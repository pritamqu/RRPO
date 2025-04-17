from transformers import AutoConfig
import os
import torch
import random
import numpy as np
import torch
from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)



def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # newly added
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        




def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0


def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side + 1, dtype=torch.bool
        )
    else:
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side, dtype=torch.bool
        )
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset(
        (num_tokens_per_side, num_tokens_per_side), image_size
    )
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset - 1 : -1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0) - 1
    return attention_mask, position_ids


def prepare_multimodal_data_signed(
    input_ids,  # pyre-fixme[2]
    labels,  # pyre-fixme[2]
    signs,
    attention_mask,  # pyre-fixme[2]
    image_sizes,  # pyre-fixme[2]
    image_token_len=576,  # pyre-fixme[2]
    image_aux_token_len_list=[192 * 192],  # pyre-fixme[2]
    max_length=2048,  # pyre-fixme[2]
):
    input_ids_im_replaced = []
    labels_im_replaced = []
    signs_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
    base_image_token_len_per_side = int(image_token_len**0.5)
    image_aux_token_len_per_side_list = [
        int(image_aux_token_len_per_side**0.5)
        for image_aux_token_len_per_side in image_aux_token_len_list
    ]
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        assert num_images == 1, num_images
        image_size = image_sizes[batch_idx]

        image_token_indices = (
            [-1]
            + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_signs_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []

        cur_labels = labels[batch_idx]
        cur_signs = signs[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(
                cur_input_ids[
                    image_token_indices[i] + 1 : image_token_indices[i + 1] + 1
                ]
            )
            cur_labels_im_replaced.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_signs_im_replaced.append(
                cur_signs[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_attention_mask_im_replaced.append(
                cur_attention_mask[
                    image_token_indices[i] + 1 : image_token_indices[i + 1]
                ]
            )
            cur_position_ids_im_replaced.append(
                torch.arange(
                    index,
                    index + image_token_indices[i + 1] - (image_token_indices[i] + 1),
                    dtype=torch.long,
                    device=cur_input_ids.device,
                )
            )
            index += image_token_indices[i + 1] - (image_token_indices[i] + 1)

            if i < len(image_token_indices) - 2:
                num_tokens_per_side = int(image_token_len**0.5)
                image_token_len_with_newline = image_token_len + num_tokens_per_side
                cur_input_ids_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline - 1,),
                        0,
                        device=cur_input_ids.device,
                        dtype=cur_input_ids.dtype,
                    )
                )
                cur_labels_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline,),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                cur_signs_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline,),
                        IGNORE_INDEX,
                        device=cur_signs.device,
                        dtype=cur_signs.dtype,
                    )
                )

                cur_im_attention_mask, cur_im_position_ids = prepare_image_info(
                    image_size, image_token_len, newline=True
                )

                for aux_i, image_aux_token_len_per_side in enumerate(
                    image_aux_token_len_per_side_list
                ):
                    assert image_aux_token_len_per_side >= base_image_token_len_per_side
                    num_base_crops_per_aux_side = (
                        image_aux_token_len_per_side // base_image_token_len_per_side
                    )

                    cur_im_aux_attention_mask, _ = prepare_image_info(
                        image_size, image_aux_token_len_per_side**2
                    )
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.view(
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                    )
                    cur_im_aux_attention_mask = (
                        cur_im_aux_attention_mask.permute(0, 2, 1, 3)
                        .contiguous()
                        .flatten(0, 1)
                        .flatten(1, 2)
                    )
                    cur_im_aux_attention_mask[
                        cur_im_aux_attention_mask.sum(dim=1) == 0
                    ] = True
                    im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
                cur_im_position_ids += index

                if cur_attention_mask[image_token_indices[i + 1]]:
                    cur_attention_mask_im_replaced.append(cur_im_attention_mask)
                    cur_position_ids_im_replaced.append(
                        cur_im_position_ids.to(torch.long)
                    )
                    index = cur_im_position_ids.max() + 1
                else:
                    num_tokens_per_side = int(image_token_len**0.5)
                    image_token_len_with_newline = image_token_len + num_tokens_per_side
                    cur_attention_mask_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_attention_mask.device,
                            dtype=cur_attention_mask.dtype,
                        )
                    )
                    cur_position_ids_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_input_ids.device,
                            dtype=torch.long,
                        )
                    )

        input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
        labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
        signs_im_replaced.append(torch.cat(cur_signs_im_replaced))
        attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
        position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))

    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_signs = [x[0:max_length] for x in signs_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = torch.stack(new_input_ids)
    new_labels = torch.stack(new_labels)
    new_signs = torch.stack(new_signs)
    new_attention_mask = torch.stack(new_attention_mask)
    new_position_ids = torch.stack(new_position_ids)
    im_aux_attention_masks_list = [
        torch.stack(im_aux_attention_masks)
        for im_aux_attention_masks in im_aux_attention_masks_list
    ]
    return (
        new_input_ids,
        new_labels,
        new_signs,
        new_attention_mask,
        new_position_ids,
        im_aux_attention_masks_list,
    )
