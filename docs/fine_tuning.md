# References

## Fine Tuning
1. [Hugging Face - Supervised Fine-Tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)

2. [Medium - LoRA for Fine-Tuning LLMs explained with codes and example](https://medium.com/data-science-in-your-pocket/lora-for-fine-tuning-llms-explained-with-codes-and-example-62a7ac5a3578)

3. [Medium - Finetuning LLMs using LoRA](https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48)
4. [Training and evalutation with pytorch , tensorflow and Keras](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/training.ipynb#scrollTo=KEmegdROhBXF)
5. [Fine-tuning phi-2](https://colab.research.google.com/github/prsdm/fine-tuning-llms/blob/main/Fine-tuning-phi-2-model.ipynb#scrollTo=yQAVFua5xIP0)
6. [Best and detailed guide to supervised fine tuning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb)
7. [Loading of LORA](https://discuss.huggingface.co/t/how-to-properly-load-the-peft-lora-model/51644/4)
8. [TEXT to SQL LLM Finetuning](https://medium.com/@jayeshchouhan826/the-ultimate-guide-to-fine-tuning-large-language-models-with-hugging-face-c971e588bf02)
9. [Perfect Finetuning Example](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing)


## Research Paper

1. [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)


## Supervised Finetuning Details

Fine-tuning is a crucial aspect of the machine learning training process. Often, there are multiple tasks at hand, and we need to fine-tune our base model to adapt to these specific tasks.

However, fully fine-tuning the entire model comes with challenges. It is resource-intensive, and it can overwrite weights learned during the initial training phase, potentially causing the model to forget previously acquired knowledge (a phenomenon known as catastrophic forgetting).

To address this, we can use a strategy called Parameter-Efficient Fine-Tuning (PEFT). This approach leverages techniques like LoRA (Low-Rank Adaptation) or QLoRA, which introduce small adapter layers on top of existing model components such as multi-head attention, query, and key weights. These adapters add new parameters specifically for the task at hand, while keeping the original model weights frozen. Essentially, you're training only these additional layers, making the process more efficient and preserving the original knowledge of the model.

And there is a recent popular way of fine tuning the model is using unsloth, checkout this repository https://github.com/unslothai/unsloth

In the following sections, we'll explore different examples of how PEFT can be applied to various tasks.

**1) Text Generation Task Fine-tuning**

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


**IMPORTANT** - **Essentially language mode tranining just take TEXT , now it is matter of how you format the text that defines different tasks, now generally when you are performing a finetuning on the base models, you are free to use any chat template or format for dataset and alpaca from stanford is popular choice. But beware, when finetuning the instruct based model, please follow exactly the model template as it is otherwise it will confuse the model**.

so, basically we have three options for the datasets, 

1) Select the dataset format recommended by hugging face
https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support

![image](https://github.com/user-attachments/assets/16426ef0-502b-4673-b052-3f4dbdd91a1a)

where SFTTrainer internally call Tokenizer's apply_chat_template method to convert to text like below and you don't need to worry about the formatting of token part. 

![image](https://github.com/user-attachments/assets/b9048acb-c03d-4367-80a6-9b8b0e33f0d6)

2) Provide a dataset containing only **text column**

![image](https://github.com/user-attachments/assets/da9e168a-a65a-4791-a456-7ae5d88ba797)


3) Have a two to three columns of data, but provide a formatting function which essentially combine multiple column in single
   text field

   ![image](https://github.com/user-attachments/assets/0cbb7af0-7fd9-4965-ba5f-4788a86411c0)

   Below format is new, never tried but should work fine

   ![image](https://github.com/user-attachments/assets/6f867d93-c81c-48e9-9e0d-86ed679b3f95)

4) Unsloth provides multiple chat formatting check here - https://docs.unsloth.ai/basics/chat-templates so, eventually applying these template you should specify "text" field in SFT trainer
   as above and eventually model will take care of the things

   ( Note - Padding during the fine-tuning is necessary in multiple cases so please beware of that. And make "Add Generation" to False while training ) 


**2) Classification Task**
Reference way can be used for 
1. [LLM Classification With Zero shot](https://stackoverflow.com/questions/76372007/trying-to-install-guanaco-pip-install-guanaco-for-a-text-classification-model/76372390#76372390) 
2. [LLM Classification With Trainer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb)
3. [Best Classification example exactly resembling above text generation fine tuning](https://www.datacamp.com/tutorial/fine-tuning-llama-3-1)

**3) Regression Task**
Generally LLM used for text related tasks, but it might/can be adapt to regression scenario by putting Linear layer on top on BERT model, 
[Linear regression using BERT](https://github.com/tomerlieber/Multi-Label-Emotion-Regression/blob/main/Emotion%20Regression%20using%20BERT.ipynb)


