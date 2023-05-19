from tempfile import NamedTemporaryFile
from typing import Any
import traceback
import sys

import streamlit as st
from streamlit_option_menu import option_menu
import torch


from instance import Instance, create_instance, get_instance_type
from model import Model
from explainer import Explainer, create_explainer
from utils import get_classification_label


def init_session_state() -> None:
    # Flags / Counters
    flags_and_counters = {
        "model_validity": False,
        "instance_validity": False,
        "output_view_index": 0,
    }
    for k, v in flags_and_counters.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Global Objects
    handlers = {
        "model": None,
        "instance": None,
        "explainer": None,
        "prediction": None,
        "explanation": None,
    }
    for k, v in handlers.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Page Configs
    st.set_page_config(page_title="Model Explainer Dashboard", page_icon="👾")
    css_config = {
        "max_width": 1100,
        "padding_top": 1,
        "padding_right": 2,
        "padding_left": 2,
        "padding_bottom": 1,
        "color": "#f8f9fa",
        "background-color": "#343a40",
    }
    st.markdown(
        f"""
<style>
    .appview-container .main .block-container{{
        max-width: {css_config['max_width']}px;
        padding-top: {css_config['padding_top']}rem;
        padding-right: {css_config['padding_right']}rem;
        padding-left: {css_config['padding_left']}rem;
        padding-bottom: {css_config['padding_bottom']}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def title_and_menu() -> str:
    st.write("# Model Explainer Dashboard")
    page = option_menu(
        None,
        ["Upload", "Explain"],
        icons=["upload", "search"],
        default_index=0,
        orientation="horizontal",
    )
    return page


def next() -> None:
    assert st.session_state["explanation"] is not None
    st.session_state["output_view_index"] = (st.session_state["output_view_index"] + 1) % len(
        st.session_state["explanation"]
    )


def prev() -> None:
    assert st.session_state["explanation"] is not None
    st.session_state["output_view_index"] = (st.session_state["output_view_index"] - 1) % len(
        st.session_state["explanation"]
    )


def fetch_prediction_and_explanation() -> None:
    try:
        with st.spinner(text="Fetching model prediction..."):
            st.session_state["prediction"] = st.session_state["model"].predict(
                st.session_state["instance"].preprocess()
            )
    except Exception as e:
        error_message = f"Model prediction failed :no_entry_sign:  \n" f"{e}"
        st.error(error_message)
    try:
        with st.spinner(text="Fetching model explanation..."):
            st.session_state["explanation"] = st.session_state["explainer"].explain()
    except Exception as e:
        error_message = f"Model explanation failed :no_entry_sign:  \n" f"{e}"
        st.error(error_message)


def show_explanation() -> None:
    while True:
        try:
            predicted_val, predicted_idx = torch.sort(
                st.session_state["prediction"], descending=True
            )
            break
        except:
            st.stop()
    target_label = st.session_state["instance"].target

    if type(st.session_state["explanation"][0]) is tuple:
        explanation_image, explanation_label = (
            list(zip(*st.session_state["explanation"]))[0],
            list(zip(*st.session_state["explanation"]))[1],
        )
    else:
        explanation_image, explanation_label = st.session_state["explanation"], None

    outputs = st.columns(2)
    outputs[0].write(f'**Original {get_instance_type(st.session_state["instance"])}**')
    outputs[0].image(st.session_state["instance"].preview(), use_column_width=True)

    outputs[1].write("**Explanation**")
    outputs[1].image(
        explanation_image[st.session_state["output_view_index"]], use_column_width=True
    )

    outputs[0].write("Ground Truth Label")
    outputs[0].code(f"{get_classification_label(target_label)}")
    outputs[1].write("Predicted Label")
    if predicted_idx[0][0] == target_label:
        outputs[1].code(f"{get_classification_label(predicted_idx[0][0])} ✅")
    else:
        outputs[1].code(f"{get_classification_label(predicted_idx[0][0])} 🚫")

    if explanation_label is not None:
        assert (
            explanation_label[st.session_state["output_view_index"]]
            == predicted_idx[0][st.session_state["output_view_index"]]
        )
        outputs[1].write(
            f"Current Explanation Label  ( {st.session_state['output_view_index'] + 1} / {len(st.session_state['explanation'])} )"
        )
        outputs[1].code(
            f'{get_classification_label(explanation_label[st.session_state["output_view_index"]]) + ":":<20} {predicted_val[0][st.session_state["output_view_index"]]:.5f}'
        )

    if len(st.session_state["explanation"]) > 1:
        buttons = st.columns(2)
        buttons[1].button("Next ➡️", on_click=next, use_container_width=True)
        buttons[0].button("⬅️ Previous", on_click=prev, use_container_width=True)


def upload_page() -> bool:
    st.write("Select a model and an instance to be explained.")

    model_expander = st.expander(label="Upload Model Checkpoint", expanded=True)
    model_file = model_expander.file_uploader("Model checkpoint file", type=["pth"])

    instance_expander = st.expander(label="Upload Input Instance", expanded=True)
    instance_type = instance_expander.radio(
        "Input instance type", ["Image", "Text", "Tabular"], horizontal=True, disabled=True
    )
    instance_file = instance_expander.file_uploader("Input instance file")
    instance_target = instance_expander.number_input(
        "Ground truth target", min_value=0, max_value=1000, step=1
    )

    # Load the input model checkpoint
    if model_file is not None:
        try:
            with NamedTemporaryFile(dir=".", suffix="pth") as f:
                f.write(model_file.getbuffer())
                st.session_state["model"] = Model(f.name)
                st.session_state["model"].load_model(f.name)
                st.session_state["model_validity"] = True
        except Exception as e:
            st.session_state["model_validity"] = False
            st.error(f"This checkpoint appears to be invalid :no_entry_sign:")
            st.error(e)

    # Preprocess the input instances
    if instance_file is not None:
        try:
            st.session_state["instance"] = create_instance(instance_type)(
                instance_file, instance_target
            )
            preview = st.session_state["instance"].preview()

            instance_expander.image(preview, caption="Input Instance Preview")
            st.session_state["instance_validity"] = True
        except NotImplementedError as error:
            st.session_state["instance_validity"] = False
            st.error(error.args[0])
        except Exception as e:
            st.session_state["instance_validity"] = False
            st.error(f"This intance appears to be invalid :no_entry_sign:")
            st.error(e)

    if st.session_state["model_validity"] and st.session_state["instance_validity"]:
        st.success("Both files appear to be valid :white_check_mark:")
        return True
    else:
        return False


def explain_page() -> None:
    if not (st.session_state["model_validity"] and st.session_state["instance_validity"]):
        st.error("Please review the model checkpoint and the input instance :no_entry_sign:")
        return

    method = st.selectbox("Explanation Tool", ["LRP", "LIME", "KernelSHAP"])
    st.session_state["explainer"] = create_explainer(
        get_instance_type(st.session_state["instance"]), method
    )()(st.session_state["model"], st.session_state["instance"])

    with st.expander("Set explainer options", expanded=True):
        try:
            options = st.session_state["explainer"].set_options()
        except ValueError as error:
            st.error(error.args[0])

    is_btn_clicked = st.button("Generate Explanation", use_container_width=True)
    st.write("---")

    if is_btn_clicked:
        fetch_prediction_and_explanation()

    show_explanation()


if __name__ == "__main__":
    init_session_state()
    page = title_and_menu()
    page_names_to_funcs = {"Upload": upload_page, "Explain": explain_page}
    page_names_to_funcs[page]()
