---
title: Hierarchical Softmax
date: 2025-05-31 17:58:46
tags: NLP
---

## Intro

In traditional word embedding models like CBOW, we input a set of ont-hot vectors, which are the contexts of a target word. And introduce a weight matrix {% katex %}W_{V \times N}{% endkatex %} to transform the vector from a very large dim V to a smaller dim N. Then transform them back with another matrix to dim V and employ Softmax to find the right target.

However, it seems not efficient enought when the word amount scales up to very large. Because the time complexity is {% katex %}O(V){% endkatex %}.

In word2vec, a Huffman Tree was introduce to encode the words efficiently and then here comes the hierarchical softmax. The time complexity can be reduced to {% katex %}O(log(V)){% endkatex %}, which is very useful when the data is very large scale.

## How it works?

The words are encoded into a Huffman Tree according to their frequency. Considering a case, the word with a higher frequency can be encoded faster than that with a lower frequency. Then the overall weighted complexity can be optimized. Now we use CBOW model to compute a word's embedding with its contextual words.

**Definitions:**

- {% katex %}w{% endkatex %}: the target word;
- {% katex %}p^{w}{% endkatex %}: the path to the target word with the give Huffman Tree;
- {% katex %}l^{w}{% endkatex %}: the length of the above path;
- {% katex %}p^{w}_{i}{% endkatex %}, where {% katex %}1 <= i <= l^{w}{% endkatex %}: each leaf in path {% katex %}p^{w}{% endkatex %};
- {% katex %}d^{w}_{i}{% endkatex %}, where {% katex %}2 <= i <= l^{w}{% endkatex %}: {% katex %}d^{w}_{l^{w}} \in \{0, 1\}{% endkatex %} represents the encoding of the word w. The others represents the encoding of the leaves in the path. 

**Target:**

To learn the representation of each word. And optimize the loss: 

{% katex %}
\mathcal{L} = \sum_{w \in \mathcal{C}} log~p(w|\text{context}(w))
{% endkatex %}
<b></b>

To compute this: ({% katex %}l^{w} - 2{% endkatex %} times binary classification)

{% katex %}
p(w|\text{context}(w)) = \prod_{j=2}^{l^{w}} p(d^{w}_j | \textbf{X}_w, \theta^{w}_{j-1})
{% endkatex %}
<b></b>

We have:

{% katex %}
\begin{align*}
p(w|\text{context}(w)) &= \left\{\begin{matrix}
\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})  &,d^{w}_{j} = 0  \\
\\
1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1}) &,d^{w}_{j} = 1  \\
\end{matrix}\right. \\
&= \sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})^{1-d^{w}_{j}}[1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})]^{d^{w}_{j}}
\end{align*}
{% endkatex %}
<b></b>

And the target loss can be formulated as:

{% katex %}
\begin{align*}
\mathcal{L} &= \sum_{w \in \mathcal{C}}log~p(w|\text{context}(w))\\
 &= \sum_{w \in \mathcal{C}} log \prod^{l^{w}}_{j=2} \left\{ \sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})^{1-d^{w}_{j}} [1 - \sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})]^{d^{w}_{j}} \right\}\\
 &= \sum_{w \in \mathcal{C}} \sum^{l^{w}}_{j=2} \left\{ (1-d^{w}_{j})log[\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})] + d^{w}_{j}log[1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})] \right\}
\end{align*}
{% endkatex %}
<b></b>

Approximately, optimizing the loss for each item:

{% katex %}
\begin{align*}
\mathcal{L}(w, j) &=  (1-d^{w}_{j})log[\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})] + d^{w}_{j}log[1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1})]\\
\end{align*}
{% endkatex %}
<b></b>

To optimize that, we find the partial derivative for {% katex %}\textbf{X}_{w}{% endkatex %} and {% katex %}\theta^{w}_{j-1}{% endkatex %}, respectively.

{% katex %}
\begin{align*}
\frac{\partial \mathcal{L}(w, j)}{\partial \theta^{w}_{j-1}} &=  [1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1}) - d^{w}_{j}]\textbf{X}^{T}_{w}
\end{align*}
{% endkatex %}
<b></b>

{% katex %}
\begin{align*}
\frac{\partial \mathcal{L}(w, j)}{\partial \textbf{X}_w} &=  [1-\sigma({\textbf{X}^{T}_{w}}\theta^{w}_{j-1}) - d^{w}_{j}]\theta^{w}_{j-1}
\end{align*}
{% endkatex %}
<b></b>

Then, to update the representation of each context word, we add the gradient to each of them like:

{% katex %}
\begin{align*}
v(\tilde{w}) &\leftarrow v(\tilde{w}) + \eta\sum^{l^w}_{j=2}\frac{\partial \mathcal{L}(w, j)}{\partial \textbf{X}_w}\\
&\text{where} ~ \tilde{w} \in \text{context}(w).
\end{align*}
{% endkatex %}
<b></b>

By this way, we can learn the word embeddings by transforming the Softmax into a process of multi-binary classification problem with a computational overhead {% katex %}O(log(V)){% endkatex %}.