import gradio as gr
import os
import time

def greet_and_show_image(name, image):
    # Here, we simply return the greeting and the image.
    # You can process the image as needed.
    greeting = "Hello " + name + "!!"
    return greeting, image

def launch_server():
    # Launch the external server script with the required parameters
    # server_command = "lmdeploy serve api_server OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
    # os.system(server_command)
    # time.sleep(60)
    command = "python /root/wangqun/layoutocr_magic_llava/gradio_web_server.py  --controller-url http://0.0.0.0:23333/v1 --concurrency-count 10 --model-list-mode reload"
    # run the command in the `internvl_chat_llava` folder
    # command = "python  /root/wangqun/layoutocr_magic_llava/gradio_web_server --controller http://localhost:23333 --model-list-mode reload"
    os.system(command)


# Define the Gradio interface with both text and image inputs.
iface = gr.Interface(
    fn=greet_and_show_image,
    inputs=[gr.Textbox(label="Enter Your Name"),
            gr.Image(label="Upload Your Image")],
    outputs=[gr.Text(label="Greeting"), gr.Image(label="Your Image")]
)

# Optional: you can launch the server in the background when initializing the Gradio app
# launch_server()

# # Launch the Gradio app
iface.launch(share=True)
