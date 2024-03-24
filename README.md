# CCoT 🧩 🧠
Official Codebase for the Paper "Compositional Chain-of-Thought Prompting for Large Multimodal Models"
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

### 💻 Setup
---
Note that because our method is a zero-shot prompting method, there is ample flexibility when applying it to your particular model and use case. As such, you may find it easier to simply use the general methodology described in our paper with a different prompt, implementation, and evaluation methodology.

### LLaVA-1.5-13b
1. First, clone the official **LLaVA** [Repository](https://github.com/haotian-liu/LLaVA).
```bash
git clone https://github.com/haotian-liu/LLaVA.git
```
2. Follow the basic installation steps outlined in the repository.
3. Complete the *Evaluation* setup outlined in the repository.
4. Replace the corresponding scripts (both Python or Bash scripts where necessary) with those in our repository here.

## GPT-4V

## InstructBLIP-13b

## Sphinx

