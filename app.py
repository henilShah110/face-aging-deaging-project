import time
import gradio as gr

from src.inference import (
    load_generator,
    process_image,
    device
)

print("Loading Models...")

age_generator = load_generator(
    "checkpoints/latest.pth",
    "age"
)

deage_generator = load_generator(
    "checkpoints/latest.pth",
    "deage"
)

print("Models Loaded!")


def generate(image, mode, progress=gr.Progress()):
    progress(0.0, desc="Preparing...")
    # choose generator


    if image is None:
        return (
            None,
            "--",
            "Please upload an image."
        )

    try:

        generator = (
            age_generator
            if mode == "Age"
            else deage_generator
        )   
        progress(0.2, desc="Loading generator...")
        start = time.time()

        progress(0.4, desc="Processing Image...")

        result, _, success = process_image(
            image,
            generator
        )

        progress(1.0, desc="Done!")
        elapsed = time.time() - start

        if not success:

            return (
                None,
                "--",
                "No face detected."
            )

        return (
            result,
            f"{elapsed:.3f} sec",
            "Success"
        )

    except Exception as e:

        return (
            None,
            "--",
            str(e)
        )


with gr.Blocks (title="Face Aging / De-Aging",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }

        footer {
            display: none;
        }

        .gr-button-primary {
            font-size: 18px;
        }"""
) as demo:
    gr.Markdown(
        """
# Face Aging / De-Aging using CycleGAN

AI-powered facial age transformation using **PyTorch, CycleGAN and MediaPipe**.
"""
    )

    with gr.Row():

        with gr.Column(scale = 1):

            input_image = gr.Image(
                type="numpy",
                label="Input Image",
                height=400
            )

            mode = gr.Radio(
                ["Age", "De-age"],
                value="Age",
                label="Mode"
            )

            generate_btn = gr.Button(
                "Generate Image",
                variant="primary"
            )

        with gr.Column(scale = 1):

            output_image = gr.Image(
                type="numpy",
                label="Generated Image",
                height=400,
                interactive=False
            )
            with gr.Row():
                inference_time = gr.Textbox(
                    label="Inference Time",
                    interactive=False
                )

                device_box = gr.Textbox(
                    label="Device",
                    value=str(device).upper(),
                    interactive=False
                )

                status_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )


    generate_btn.click(
        fn=generate,
        inputs=[
            input_image,
            mode
        ],
        outputs=[
            output_image,
            inference_time,
            status_box
        ]
    )

    clear_btn = gr.Button("Clear")
    clear_btn.click(
        lambda: (
            None,
            "Age",
            None,
            "",
            "Ready"
        ),
        outputs=[
            input_image,
            mode,
            inference_time,
            status_box
        ]
    )


    gr.Markdown("## Example Images")

    with gr.Row():

        example1 = gr.Image(
            value="examples/young1.jpg",
            label="Young Female",
            interactive=False,
            height=180
        )

        example2 = gr.Image(
            value="examples/young2.jpg",
            label="Young Male",
            interactive=False,
            height=180
        )

    with gr.Row():

        example3 = gr.Image(
            value="examples/old1.jpg",
            label="Old Male",
            interactive=False,
            height=180
        )

        example4 = gr.Image(
            value="examples/old2.jpg",
            label="Old Female",
            interactive=False,
            height=180
        )

    gr.Markdown(
        """
---

**Developed by Henil Shah**

PyTorch • CycleGAN • MediaPipe • Gradio
"""
    )

    with gr.Accordion("How it Works", open=False):
        gr.Markdown("""
        1. Detects the face using MediaPipe.

        2. Crops and aligns the face.

        3. Runs the CycleGAN generator.

        4. Blends the generated face back into the image.

        5. Displays the result with a comparison slider.
        """)


demo.queue()

if __name__ == "__main__":
    demo.launch(
        share=True
    )
