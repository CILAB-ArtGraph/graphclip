from open_clip import SimpleTokenizer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
import torch
import numpy as np


def custom_tokenize(sentence: str, base_tokenizer: SimpleTokenizer) -> list[int]:
    ans = [base_tokenizer.sot_token_id]
    for token in sentence.split(" "):
        transformed = base_tokenizer(token).tolist()[0]
        transformed = [val for val in transformed if val != 0]
        ans += transformed[1:-1]
    ans += [base_tokenizer.eot_token_id]
    return ans


def interpret_sentence(
    model: torch.nn.Module,
    sentence: str,
    tokenizer: SimpleTokenizer,
    min_len: int = 77,
    label=1,
    device="cuda",
    plot: bool = False,
) -> np.array:
    token_reference = TokenReferenceBase(reference_token_idx=0)
    lig = LayerIntegratedGradients(model, model.token_embedding)
    text = sentence[0].split()
    vis_data_records_ig = []

    indexed = tokenizer(sentence).to(device)
    input_indices = indexed.clone()
    model.zero_grad()
    pred = model(indexed).sigmoid().item()
    pred_ind = round(pred)

    reference_indices = token_reference.generate_reference(
        min_len, device=device
    ).reshape(input_indices.size())

    attributions, delta = lig.attribute(
        input_indices, reference_indices, n_steps=100, return_convergence_delta=True
    )
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    real_attributions = attributions[1 : 1 + len(text)]
    x = torch.from_numpy(real_attributions)
    raw_input_ids = text

    if plot:
        # storing couple samples in an array for visualization purposes
        vis_data_records_ig.append(
            visualization.VisualizationDataRecord(
                word_attributions=x,
                pred_prob=pred,
                pred_class=pred_ind,
                true_class=label,
                attr_class=1,
                attr_score=attributions.sum(),
                raw_input_ids=raw_input_ids,
                convergence_score=delta,
            )
        )
        visualization.visualize_text(vis_data_records_ig)
    return x
