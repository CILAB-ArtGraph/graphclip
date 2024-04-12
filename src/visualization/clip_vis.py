import torch
import streamlit as st
from src.experiment import CLIPRun
from src.utils import load_ruamel
from PIL import Image
from src.models.explainer import CLIPExplainer
from src.data import DataDict
from typing import Any
from src.visualization.utils import SessionStateKey


@st.cache_resource
def _init_run():
    return CLIPRun(
        load_ruamel("./configs/baselines/normal_clip_fine_tuning_mistral.yaml")
    )


def get_img_path(run: CLIPRun, case_id: int):
    """
    Returns the image path by the case id
    """
    img_dir = run.test_loader.dataset.img_dir
    img_path = run.test_loader.dataset.dataset.iloc[case_id, 0]
    return f"{img_dir}/{img_path}"


def get_style_by_case_id(run: CLIPRun, case_id: int):
    """
    Returns the style by case id
    """
    return run.test_loader.dataset.dataset.iloc[case_id, 1]


def get_prediction(run: CLIPRun, case_id: int) -> dict[str, Any]:
    """
    Returns the prediction by case id
    """
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
    st.session_state[SessionStateKey.IDX2CLASS] = idx2class
    st.session_state[SessionStateKey.CLASS2IDX] = class2idx
    return {
        SessionStateKey.LOGITS: prediction,
        SessionStateKey.IMG_FEATS: img_feats,
        SessionStateKey.TXT_FEATS: class_feats,
    }


def store_img_vis_session_state(img, img_pth, caption="", use_column_with=False):
    "Stores the session state for the basic image visualization"
    state = {
        SessionStateKey.CASE_IMG: {
            SessionStateKey.ST_IMG: {
                SessionStateKey.IMG: img,
                SessionStateKey.CAPTION: caption,
                SessionStateKey.USE_COL_WIDTH: use_column_with,
            },
            SessionStateKey.IMG_PTH: img_pth,
        }
    }
    st.session_state.update(state)


def restore_img_vis_session_state():
    "Restores the img visualization when other buttons are clicked if it had been clicked previously"
    state = st.session_state.get(SessionStateKey.CASE_IMG, {})
    if not state:
        return None
    img_kwargs = state.get(SessionStateKey.ST_IMG)
    st.image(**img_kwargs)
    return state.get(SessionStateKey.IMG_PTH)


def store_img_exp_session_state(imgs, caption):
    """
    Stores the info for image explanations
    """
    state = {
        SessionStateKey.IMG_EXP: {
            SessionStateKey.ST_IMG: {
                SessionStateKey.IMG: imgs,
                SessionStateKey.USE_COL_WIDTH: True,
            },
            SessionStateKey.TXT_BOX: caption,
        },
    }
    st.session_state.update(state)


def restore_img_exp_session_state():
    """
    Restores the information about image explanation when clicking on ther buttons
    """
    state = st.session_state.get(SessionStateKey.IMG_EXP, {})
    if not state:
        return
    st.image(**state.get(SessionStateKey.ST_IMG))
    st.write(state.get(SessionStateKey.TXT_BOX))


def main():
    st.title("CLIP Explainer")

    # run and explainer are essential
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

    # handle the image visualization
    if st.button("Visualize"):
        img_style = get_style_by_case_id(run=run, case_id=case_id)
        img_pth = get_img_path(run=run, case_id=case_id)
        img = Image.open(img_pth).convert("RGB")
        caption = f"GT style: {img_style}"

        st.image(img, use_column_width=True, caption=caption)

        # store the session state
        store_img_vis_session_state(
            img=img, img_pth=img_pth, caption=caption, use_column_with=True
        )
    elif SessionStateKey.CASE_IMG in st.session_state:
        img_pth = restore_img_vis_session_state()

    # handles the image explanation
    if st.button("Explain image"):
        if not case_id:
            st.error("Please choose an instance before!")
        out = get_prediction(run, case_id)
        pred_idx = out[SessionStateKey.LOGITS].argmax().cpu().item()
        pred_class = st.session_state[SessionStateKey.IDX2CLASS][pred_idx]
        overlayed_img = explainer.explain_image(
            img_path=img_pth,
            model=run.model,
            text_reference_feats=out[SessionStateKey.TXT_FEATS],
            target=pred_idx,
            overlayed=True,
        )
        st.image(
            [img_pth, overlayed_img],
            use_column_width=True,
        )
        exp_caption = f"Explaination of the image for class {pred_class}"
        st.write(exp_caption)
        store_img_exp_session_state(imgs = [img_pth, overlayed_img], caption=exp_caption)
    elif SessionStateKey.IMG_EXP in st.session_state:
        restore_img_exp_session_state()

    if st.button("Explain text"):
        if not case_id:
            st.error("Please choose an instance before!")


if __name__ == "__main__":
    main()
