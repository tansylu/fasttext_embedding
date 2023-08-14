import gradio as gr
from process_user_input import match 
# with gr.Blocks() as demo:
#     name = gr.Textbox(label="Name")
#     output = gr.Textbox(label="Output Box")
#     greet_btn = gr.Button("Greet")
#     greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
   
#on first run, run the embed_food_emissions file to obtain embeddings for the reference database
demo = gr.Interface(fn=match,
                     inputs="text", 
                    
                    outputs=[         # List of output fields
                            gr.outputs.Textbox(label="Product Type Name"),
                            gr.outputs.Textbox(label="Product Type"),
                            gr.outputs.Textbox(label="Carbon Footprint"),
                            gr.outputs.Textbox(label="Unit"),
                            gr.outputs.Textbox(label="Country Code")
                        ],
                    title="Match your food item to a carbon emissions category")
demo.launch(share=True)  