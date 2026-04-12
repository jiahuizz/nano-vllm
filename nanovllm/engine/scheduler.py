from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # GDN state slot management
        self.has_gdn = hasattr(config.hf_text_config, 'layer_types')
        if self.has_gdn:
            max_gdn_seqs = min(config.max_num_seqs, 128)
            self.gdn_free_slots: deque[int] = deque(range(max_gdn_seqs))

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Unified token-budget scheduler with chunked prefill.

        Priority: decode first, then use remaining budget for prefill chunks.
        Returns (scheduled_seqs, is_prefill).
        """
        # Phase 1: Schedule decode for running sequences that have completed prefill
        decode_seqs = []
        prefill_seqs = []
        num_seqs = 0

        running_snapshot = list(self.running)
        for seq in running_snapshot:
            if num_seqs >= self.max_num_seqs:
                break
            # Sequence has completed prefill if num_cached_tokens >= num_prompt_tokens
            if seq.num_cached_tokens >= seq.num_prompt_tokens:
                # Decode: needs 1 new token
                while not self.block_manager.can_append(seq):
                    if self.running:
                        victim = self.running.pop()
                        if victim != seq:
                            self.preempt(victim)
                        else:
                            self.preempt(seq)
                            break
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    decode_seqs.append(seq)

        # If we have decode work, do it (decode priority)
        if decode_seqs:
            return decode_seqs, False

        # Phase 2: Schedule prefill (chunked)
        token_budget = self.max_num_batched_tokens
        num_seqs = 0

        # 2a: Continue prefill for running sequences that haven't finished prefill
        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs or token_budget <= 0:
                break
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                remaining = seq.num_prompt_tokens - seq.num_cached_tokens
                chunk = min(remaining, token_budget)
                seq._prefill_chunk_size = chunk
                num_seqs += 1
                token_budget -= chunk
                prefill_seqs.append(seq)

        # 2b: Start new sequences from waiting queue
        while self.waiting and num_seqs < self.max_num_seqs and token_budget > 0:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            if self.has_gdn and not self.gdn_free_slots:
                break

            remaining = seq.num_prompt_tokens - seq.num_cached_tokens
            chunk = min(remaining, token_budget)
            seq._prefill_chunk_size = chunk

            self.block_manager.allocate(seq)
            if self.has_gdn:
                seq.gdn_state_idx = self.gdn_free_slots.popleft()
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

            num_seqs += 1
            token_budget -= chunk
            prefill_seqs.append(seq)

        if prefill_seqs:
            return prefill_seqs, True

        # Fallback: should not reach here if there's work to do
        assert False, "Scheduler has work but couldn't schedule anything"

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        if self.has_gdn and seq.gdn_state_idx >= 0:
            self.gdn_free_slots.appendleft(seq.gdn_state_idx)
            seq.gdn_state_idx = -1
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | None, is_prefill: bool):
        """Process results. For prefill: update num_cached_tokens (and append
        the first completion token when prefill finishes). For decode: append token."""
        if is_prefill:
            for i, seq in enumerate(seqs):
                chunk = getattr(seq, '_prefill_chunk_size', seq.num_prompt_tokens - seq.num_cached_tokens)
                seq.num_cached_tokens += chunk
                # When prefill completes, the model also sampled a first token — append it
                if seq.num_cached_tokens >= seq.num_prompt_tokens and token_ids is not None:
                    token_id = token_ids[i]
                    seq.append_token(token_id)
                    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        seq.finished_time = perf_counter()
                        self.block_manager.deallocate(seq)
                        if self.has_gdn and seq.gdn_state_idx >= 0:
                            self.gdn_free_slots.appendleft(seq.gdn_state_idx)
                            seq.gdn_state_idx = -1
                        self.running.remove(seq)
        else:
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    seq.finished_time = perf_counter()
                    self.block_manager.deallocate(seq)
                    if self.has_gdn and seq.gdn_state_idx >= 0:
                        self.gdn_free_slots.appendleft(seq.gdn_state_idx)
                        seq.gdn_state_idx = -1
                    self.running.remove(seq)
