# CCoT 🧩 🧠
Official Codebase for the Paper "Compositional Chain-of-Thought Prompting for Large Multimodal Models" (\*updates still in progress \*)
---
We present **CCoT**, a novel **C**compositional **C**hain-**o**f-**T**hought prompting method that utilizes scene-graph representations in order to extract compositional knowledge from an LMM. We find that this approach not only improves LMM performance on several compositional benchmarks but also general multimodal benchmarks as well. 

<p align="center">
  <img src=images/fig1_v7.png width="500"/>
</p>

> Insert links to paper, website, and everyone's website here

### Method Description
---

<p align="center">
  <img src=images/fig2_v8.png />
</p>

The first step in our prompting method is to generate a scene graph given both the image \textit{and} textual task as context. Following this, the answer is extracted by prompting the LMM with the image, scene graph, question, and answer extraction prompt. Prompt sections unique to our method are shown in **bold** in the above figure. Incorporating the scene graph in the prompt eliminates the need for fine-tuning and prevents forgetting. Another benefit of our method is that generated SGs can describe any visual scene, therefore making CCoT generally applicable to a wider range of VL tasks. Finally, the fact that the generated scene graphs are compact linguistic representations of images makes CCoT a token-efficient prompting method. This is significant given the limited textual context lengths that LMMs often face due to processing both image and text inputs.

### 💻 Setup
---
Note that because our method is a zero-shot prompting method, there is ample flexibility when applying it to your particular model and use case. As such, you may find it easier to simply use the general methodology described in our paper with a different prompt, implementation, and evaluation methodology.

#### LLaVA-1.5-13b
1. First, clone the official **LLaVA** [Repository](https://github.com/haotian-liu/LLaVA).
```bash
git clone https://github.com/haotian-liu/LLaVA.git
```
2. Follow the basic installation steps outlined in the repository.
3. Complete the *Evaluation* setup outlined in the repository.
4. Replace the corresponding scripts (both Python or Bash scripts where necessary) with those in our repository here.

#### GPT-4V

#### InstructBLIP-13b

#### Sphinx

