import torch
from torch import distributed as dist
import torch.nn.functional as F

def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)

# modified from TDPO: https://github.com/Vance0124/Token-level-Direct-Preference-Optimization
class RRPO(object):
    def __init__(self, 
                 alpha, 
                 beta, 
                 ) -> None:
        self.alpha=alpha
        self.beta=beta

    def __call__(self, align_dict, ref_dict, rank, world_size):
        chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps\
                = self.concatenated_forward(align_dict, ref_dict)
        losses, chosen_rewards, rejected_rewards = self.loss(chosen_logps_margin, rejected_logps_margin,
                                                                 chosen_position_kl, rejected_position_kl,
                                                                 beta=self.beta, alpha=self.alpha)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_rewards = all_gather_if_needed(chosen_rewards, rank, world_size)
        rejected_rewards = all_gather_if_needed(rejected_rewards, rank, world_size)
        reward_accuracies = all_gather_if_needed(reward_accuracies, rank, world_size)

        metrics = {}
        train_test="train"
        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), rank, world_size)
        all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), rank, world_size)

        metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
        metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
        metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

        policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), rank, world_size)
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), rank, world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), rank, world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics


    def concatenated_forward(self, align_dict, ref_dict):
        all_logits=align_dict['logits']
        reference_all_logits=ref_dict['logits']
        targets=align_dict['targets']
        signs=align_dict['signs']
        half_batch=all_logits.shape[0]//2
        all_logps_margin, phrase_logps_margin, all_position_kl, all_logps = self.get_batch_logps(all_logits, reference_all_logits, 
                                                                             targets, signs,
                                                                             average_log_prob=False)

        # chosen_logps_margin = all_logps_margin[:half_batch]
        # rejected_logps_margin = all_logps_margin[half_batch:]
        chosen_logps_margin = phrase_logps_margin[:half_batch] # [B, N], N is number of phrases
        rejected_logps_margin = phrase_logps_margin[half_batch:]# [B, N], N is number of phrases
        chosen_position_kl = all_position_kl[:half_batch]
        rejected_position_kl = all_position_kl[half_batch:]

        chosen_logps = all_logps[:half_batch].detach()
        rejected_logps = all_logps[half_batch:].detach()

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, \
            chosen_logps, rejected_logps

    def accumulate_logps(self, logps, signs):
        unique_signs, indices = torch.unique(signs, sorted=True, return_inverse=True)
        accumulated_logps = torch.zeros(signs.size(0), len(unique_signs) - 1, dtype=logps.dtype, device=logps.device)
        
        for i, sign in enumerate(unique_signs[1:]):
            mask = (signs == sign).float()
            accumulated_logps[:, i] = (logps * mask).sum(dim=-1)
        
        return accumulated_logps
    
    def get_batch_logps(self, logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                            signs: torch.LongTensor,
                            average_log_prob: bool = False
                            ):
        """Compute the kl divergence/log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape
        assert reference_logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        signs = signs[:, 1:].clone()
        logits = logits[:, :-1, :]
        reference_logits = reference_logits[:, :-1, :]

        loss_mask = (labels != -100)
        signs = signs.masked_fill(signs==-100, 0)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        vocab_logps = logits.log_softmax(-1)

        reference_vocab_ps = reference_logits.softmax(-1)
        reference_vocab_logps = reference_vocab_ps.log()

        per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
        per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logps_margin = per_token_logps - per_reference_token_logps

        # for phrase-level logps_margin
        # this will return log-probabilities of the phrases, [B, N], if there are N phrases
        phrase_logps_margin = self.accumulate_logps(logps_margin, signs)

        # if average_log_prob:
        #     return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
        #         (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
        #         (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # else:
        #     return (logps_margin * loss_mask).sum(-1), \
        #         (per_position_kl * loss_mask).sum(-1), \
        #         (per_token_logps * loss_mask).sum(-1)

        # calculating avg should be based on number of phrases per sample
        assert average_log_prob == False # TODO: implement this later
        return (logps_margin * loss_mask).sum(-1), \
                phrase_logps_margin, \
                (per_position_kl * loss_mask).sum(-1), \
                (per_token_logps * loss_mask).sum(-1),


    def loss(self, chosen_logps_margin: torch.FloatTensor,
                rejected_logps_margin: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                beta: float, alpha: float = 0.5, 
                ):
        """Compute the RRPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter to control the penalty on the non-preferred phrases. 
                As beta -> 0, we ignore the reference model.
                Suggested range 0.1 to 0.9. 
            alpha: Control the divergence between the model and reference model through token-wise kl divergence. 
                As alpha -> 0, we diverge far away from the reference model. 
                Suggested range 0.01 to 0.1.

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the RRPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """

        # we are calculating the margin at a phrase level
        chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin # [B, N], if there are N phrases
        chosen_rejected_logps_margin = chosen_rejected_logps_margin.sum(-1) # finally taking the sum

        # as log_ps are in [B, N] shape
        chosen_values = chosen_logps_margin.sum(-1)
        rejected_values = rejected_logps_margin.sum(-1)
        
        logits = chosen_rejected_logps_margin
        losses = -F.logsigmoid(beta * logits) + alpha * chosen_position_kl
        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()
        return losses, chosen_rewards, rejected_rewards
    
        
