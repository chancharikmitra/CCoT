# CCoT 🧩 🧠
Official Code for the Paper "Compositional Chain-of-Thought Prompting for Large Multimodal Models" (\*updates still in progress\*)
---
We present **CCoT**, a novel **C**compositional **C**hain-**o**f-**T**hought prompting method that utilizes scene-graph representations in order to extract compositional knowledge from an LMM. We find that this approach not only improves LMM performance on several compositional benchmarks but also general multimodal benchmarks as well. 

A more thorough discussion of our work can be found in our (paper)[https://arxiv.org/abs/2311.17076].

<p align="center">
  <img src=images/fig1_v7.png width="500"/>
</p>

### Method Description
---

<p align="center">
  <img src=images/fig2_v8.png />
</p>

The first step in our prompting method is to generate a scene graph given both the image *and* textual task as context. Following this, the answer is extracted by prompting the LMM with the image, scene graph, question, and answer extraction prompt. Prompt sections unique to our method are shown in **bold** in the above figure. Incorporating the scene graph in the prompt eliminates the need for fine-tuning and prevents forgetting. Another benefit of our method is that generated SGs can describe any visual scene, therefore making CCoT generally applicable to a wider range of VL tasks. Finally, the fact that the generated scene graphs are compact linguistic representations of images makes CCoT a token-efficient prompting method. This is significant given the limited textual context lengths that LMMs often face due to processing both image and text inputs.

### 💻 Setup
---
**Note** that because our method is a zero-shot prompting method and makes use of the codebase of its respective LMM, there is ample flexibility when applying it to your particular model and use case. As such, you may find it *easier* to simply use the general methodology described in our paper and outlined in our scripts with a different prompt, implementation, and evaluation methodology to suit your needs.

#### Datasets
Please retrieve all datasets from their respective official websites or repositories. We do provide the filtered .jsonl containing just the SEEDBench-Image data points in our data folder.

#### LLaVA-1.5-13b
1. First, clone the official **LLaVA** [repository](https://github.com/haotian-liu/LLaVA).
```bash
git clone https://github.com/haotian-liu/LLaVA.git
```
2. Follow the basic installation steps outlined in the repository.
3. Complete the *Evaluation* setup outlined in the repository.
4. Replace the corresponding scripts (both Python or Bash scripts where necessary) with those in our repository here.

#### GPT-4V

1. Install the openai library:
```bash
pip install openai
```
2. Set your openai key:
```bash
export OPENAI_API_KEY=
```
3. Run the script for your desired dataset.

#### InstructBLIP-13b

1. First, clone the official **LLaVA** [Repository]([https://github.com/haotian-liu/LLaVA](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)).
2. Follow the basic installation steps outlined in the repository.
3. Run the script for your desired dataset.

#### Sphinx

1. For SEEDBench and MMBench, we make use of the LLaVA codebase's setup. Simply follow the LLaVA-1.5 setup steps and replace the scripts with those of Sphinx.
2. For other datasets, follow setup instructions from the official [repository](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)
3. Run our provided script.

### 📝 Citation
---
If you found our work useful, please consider starring and citing. Thank you!
```latex
@inproceedings{Mitra,
  title={Compositional Chain of Thought Prompting for Large Multimodal Models},
  author={Mitra, Chancharik and Huang, Brandon and Darrell, Trevor and Herzig, Roei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2024}
}
```
