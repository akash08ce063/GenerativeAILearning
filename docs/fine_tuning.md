# References

## Fine Tuning
1. [Hugging Face - Supervised Fine-Tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)

2. [Medium - LoRA for Fine-Tuning LLMs explained with codes and example](https://medium.com/data-science-in-your-pocket/lora-for-fine-tuning-llms-explained-with-codes-and-example-62a7ac5a3578)

3. [Medium - Finetuning LLMs using LoRA](https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48)
4. https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb#scrollTo=KEmegdROhBXF
5. https://colab.research.google.com/github/prsdm/fine-tuning-llms/blob/main/Fine-tuning-phi-2-model.ipynb#scrollTo=yQAVFua5xIP0
6. https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb
7. https://discuss.huggingface.co/t/how-to-properly-load-the-peft-lora-model/51644/4
8. [TEXT to SQL LLM Finetuning](https://medium.com/@jayeshchouhan826/the-ultimate-guide-to-fine-tuning-large-language-models-with-hugging-face-c971e588bf02)
9. [Perfect Finetuning Example](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing)


## Research Paper

1. [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)


## Information

Fine-tuning is a crucial aspect of the machine learning training process. Often, there are multiple tasks at hand, and we need to fine-tune our base model to adapt to these specific tasks.

However, fully fine-tuning the entire model comes with challenges. It is resource-intensive, and it can overwrite weights learned during the initial training phase, potentially causing the model to forget previously acquired knowledge (a phenomenon known as catastrophic forgetting).

To address this, we can use a strategy called Parameter-Efficient Fine-Tuning (PEFT). This approach leverages techniques like LoRA (Low-Rank Adaptation) or QLoRA, which introduce small adapter layers on top of existing model components such as multi-head attention, query, and key weights. These adapters add new parameters specifically for the task at hand, while keeping the original model weights frozen. Essentially, you're training only these additional layers, making the process more efficient and preserving the original knowledge of the model.

In the following sections, we'll explore different examples of how PEFT can be applied to various tasks.

**Text Generation Task Fine-tuning**

For fine-tuning a model on text generation tasks, we can leverage Hugging Face's tools effectively. Specifically, we can use a pre-trained LLAMA model along with Parameter-Efficient Fine-Tuning (PEFT) techniques. The setup involves defining a PEFT configuration and utilizing the SFTTrainer class, which is a subclass of Hugging Face's main Trainer class.

The SFTTrainer is designed to simplify the fine-tuning process, expecting a pre-formatted dataset that aligns with Hugging Face's requirements. The advantage here is that Hugging Face provides datasets specifically formatted for SFT, meaning you don't need to handle the preprocessing manually.

Since text generation is essentially a next-word prediction task, providing the dataset in the proper format allows the SFTTrainer to handle everything automatically.
Example Datasets for Text Generation Fine-tuning
You can use any of the following datasets for text generation fine-tuning:

[Guanaco LLAMA2 (1k samples)](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
[MedQuad-PHI2 (1k samples)](https://huggingface.co/datasets/prsdm/MedQuad-phi2-1k?row=0)
[UltraChat (200k samples)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k?row=23)
These datasets are structured to be compatible with the SFTTrainer right out of the box, making the training process efficient and straightforward.

Chat Model Considerations
For chat-based models, take a look at the [Chat Templates](https://huggingface.co/blog/chat-templates) provided by Hugging Face. Large Language Models (LLMs) are often trained on various data formats, and it's beneficial to align your fine-tuning dataset with a specific format. During inference, if you maintain consistency with the format used in training, you can significantly improve prediction accuracy.

By following these guidelines, you can efficiently fine-tune a model for text generation tasks and achieve better performance tailored to your specific use case.

Let's take an example, 
so for this tutorial [SFT](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb) 

example of dataset format 

![image](https://github.com/user-attachments/assets/5c9a2179-368d-440c-9de6-b0849c6f20fe)

so we apply chat template and convert data in below format and add new property "**text**"
![image](https://github.com/user-attachments/assets/2bbb4b12-961f-4f92-b48d-faf9aed2fb36)

now, we use the name of field "text" in SFTTrainer argument for training purpose
![image](https://github.com/user-attachments/assets/a9c20cc6-b9c6-4b7e-a80e-318e04248b60)


**IMPORTANT** - Essentially language mode tranining just take TEXT , now it is matter of how you format the text that defines different tasks
