# RepE Demo

This repository contains a quick demo of implementation of [representation learning](https://www.ai-transparency.org/ "representation learning"). The demo is particularly for risk assessment in the advice for financial investments.

---

### Instructions for running:
Create a python environment and install the dependencies.
`$ pip install -r requirement.txt`

- A toy dataset for financial domain can be created using **representation_learning_notebooks/create_data.ipynb**
- Run the demo here: **representation_learning_notebooks/repe_risk.ipynb**

---

### Pros and Cons of RepE method

### **Pros**

* **Better Understanding**

  * RepE looks at high-level ideas, not just neurons.
  * This helps us see how models think about things like honesty or fairness.
  * It gives a top-down view, making complex behavior easier to study.

* **Works in Many Areas**

  * RepE can handle tasks like truth or lie detection, fairness, and more.
  * It helps both in understanding and changing model behavior.

* **Less Bias, More Scale**

  * It uses unsupervised examples, so no need for lots of labels.
  * This cuts down on bias and scales better with big models.
  * It uses simple tools like PCA to find useful directions in model space.

---

### **Cons**

* **Same Direction for All Inputs**

  * The method doesn’t change much based on the input.
  * This can lead to weaker control in some situations.

* **Takes More Computing Power**

  * Making and analyzing example pairs takes time.
  * Using it on big models can slow things down a lot.

* **Sensitive to Setup**

  * The results depend on how you choose example pairs and tasks.
  * Small changes in settings can affect how well it works.

---
