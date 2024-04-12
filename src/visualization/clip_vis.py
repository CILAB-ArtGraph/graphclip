import torch
import streamlit as st
from src.experiment import CLIPRun
from src.utils import load_ruamel
from PIL import Image
from src.models.explainer import CLIPExplainer
from src.data import DataDict
from typing import Any


@st.cache_resource
def _init_run():
    return CLIPRun(
        load_ruamel("./configs/baselines/normal_clip_fine_tuning_mistral.yaml")
    )


def get_img_path(run: CLIPRun, case_id: int):
    img_dir = run.test_loader.dataset.img_dir
    img_path = run.test_loader.dataset.dataset.iloc[case_id, 0]
    return f"{img_dir}/{img_path}"


def get_style_by_case_id(run: CLIPRun, case_id: int):
    return run.test_loader.dataset.dataset.iloc[case_id, 1]


def get_prediction(run: CLIPRun, case_id: int) -> dict[str, Any]:
    run.model = run.model.to(run.device)
    class_prompts, idx2class, class2idx = run.get_class_info()
    class_tokens = run.tokenizer(class_prompts).to(run.device)
    class_feats = run.model.encode_text(class_tokens, normalize=True)
    img_tensor = (
        run.test_loader.dataset[case_id].get(DataDict.IMAGE).unsqueeze(0).to(run.device)
    )
    img_feats = run.model.encode_image(img_tensor, normalize=True)
    prediction = img_feats @ class_feats.T
    # save to session_state
    st.session_state["idx2class"] = idx2class
    st.session_state["class2idx"] = class2idx
    return {
        "prediction": prediction,
        "img_feats": img_feats,
        "class_feats": class_feats,
    }


def main():
    st.title("CLIP Explainer")

    run = _init_run()
    explainer = CLIPExplainer(
        device=run.device,
        image_preprocess=run.test_loader.dataset.preprocess,
        tokenizer=run.tokenizer,
    )

    case_id = st.slider(
        "Select the test case",
        min_value=0,
        max_value=len(run.test_loader.dataset) - 1,
        step=1,
    )
    if st.button("Visualize"):
        img_style = get_style_by_case_id(run=run, case_id=case_id)
        img_pth = get_img_path(run=run, case_id=case_id)
        img = Image.open(img_pth).convert("RGB")
        st.image(img, use_column_width=True, caption=f"GT style: {img_style}")
        st.session_state["vis_img"] = img
        st.session_state["gt"] = img_style
        st.session_state["img_pth"] = img_pth
    elif "vis_img" in st.session_state:
        st.image(
            st.session_state.get("vis_img"),
            use_column_width=True,
            caption=f"GT style: {st.session_state.get('gt')}",
        )
        img_pth = st.session_state["img_pth"]

    if st.button("Explain image"):
        if not case_id:
            st.error("Please choose an instance before!")
        out = get_prediction(run, case_id)
        pred_idx = out["prediction"].argmax().cpu().item()
        pred_class = st.session_state["idx2class"][pred_idx]
        overlayed_img = explainer.explain_image(
            img_path=img_pth,
            model=run.model,
            text_reference_feats=out["class_feats"],
            target=out["prediction"].argmax().cpu().item(),
            overlayed=True,
        )
        st.image(
            [img_pth, overlayed_img],
            use_column_width=True,
        )
        st.write(f"Explaination of the image for class {pred_class}")

    if st.button("Explain text"):
        if not case_id:
            st.error("Please choose an instance before!")


if __name__ == "__main__":
    main()
