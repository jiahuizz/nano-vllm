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

    def schedule_decode(self) -> list[Sequence]:
        """Schedule decode for running sequences that have completed prefill."""
        decode_seqs = []
        for seq in list(self.running):
            if len(decode_seqs) >= self.max_num_seqs:
                break
            if seq.num_cached_tokens >= seq.num_prompt_tokens:
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
                    self.block_manager.may_append(seq)
                    decode_seqs.append(seq)
        return decode_seqs

    def schedule_prefill(self) -> list[Sequence]:
        """Schedule prefill chunks for waiting/partial sequences using available slots."""
        prefill_seqs = []
        token_budget = self.max_num_batched_tokens
        num_seqs = len(self.running)  # count current running as occupied slots

        # Continue prefill for running sequences that haven't finished prefill
        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs or token_budget <= 0:
                break
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                remaining = seq.num_prompt_tokens - seq.num_cached_tokens
                chunk = min(remaining, token_budget)
                seq._prefill_chunk_size = chunk
                token_budget -= chunk
                prefill_seqs.append(seq)

        # Start new sequences from waiting queue
        while self.waiting and num_seqs < self.max_num_seqs and token_budget > 0:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            if self.has_gdn and not self.gdn_free_slots:
                break

            self.block_manager.allocate(seq)  # allocate first — may set num_cached_tokens via prefix cache
            if self.has_gdn:
                seq.gdn_state_idx = self.gdn_free_slots.popleft()
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

            # Now compute chunk with post-allocate num_cached_tokens
            remaining = seq.num_prompt_tokens - seq.num_cached_tokens
            if remaining == 0:
                # Fully cached by prefix cache — no prefill needed, skip (decode will pick it up next step)
                seq._prefill_chunk_size = 0
                num_seqs += 1
                continue
            chunk = min(remaining, token_budget)
            seq._prefill_chunk_size = chunk

            num_seqs += 1
            token_budget -= chunk
            prefill_seqs.append(seq)

        return prefill_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        if self.has_gdn and seq.gdn_state_idx >= 0:
            self.gdn_free_slots.appendleft(seq.gdn_state_idx)
            seq.gdn_state_idx = -1
        self.waiting.appendleft(seq)

    def _finish_seq(self, seq: Sequence):
        """Mark a sequence as finished and release its resources."""
        seq.status = SequenceStatus.FINISHED
        seq.finished_time = perf_counter()
        self.block_manager.deallocate(seq)
        if self.has_gdn and seq.gdn_state_idx >= 0:
            self.gdn_free_slots.appendleft(seq.gdn_state_idx)
            seq.gdn_state_idx = -1
        self.running.remove(seq)

    def postprocess_decode(self, seqs: list[Sequence], token_ids: list[int]):
        """Append decoded tokens. Finished seqs are removed, freeing slots."""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                self._finish_seq(seq)

    def postprocess_prefill(self, seqs: list[Sequence], token_ids: list[int] | None):
        """Update prefill progress. When prefill completes, append the first token."""
        for i, seq in enumerate(seqs):
            chunk = getattr(seq, '_prefill_chunk_size', seq.num_prompt_tokens - seq.num_cached_tokens)
            seq.num_cached_tokens += chunk
            if seq.num_cached_tokens >= seq.num_prompt_tokens and token_ids is not None:
                token_id = token_ids[i]
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id in self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    self._finish_seq(seq)
