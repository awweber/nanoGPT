# Nanochat: Ein nano-LLM

Das hier ist eine Schritt-für-Schritt-Anleitung, um ein winziges Sprachmodell (LLM) zu bauen, das Shakespeare-ähnlichen Text generiert. Wir nutzen dabei die Transformer-Architektur, die auch in großen Modellen wie GPT-4 verwendet wird, aber in einer stark vereinfachten Form. Es basiert auf dem berühmten Shakespeare-Datensatz, der öffentlich verfügbar ist.

Das Beispiel hier basiert auf nanochat von Andrej Karpathy, das quasi das „Hello World“-Projekt für LLMs darstellt, siehe auch von Andrej Karpathys „Let's build GPT: from scratch“ (nanoGPT). Andrej Karpathy (ehemals Director of AI bei Tesla und OpenAI) hat ein Projekt erstellt, das ein GPT-Modell (Generative Pre-trained Transformer) Zeichen für Zeichen nachbaut.

## Voraussetzungen
- Grundkenntnisse in Python und PyTorch
- Installation von Python 3.8+ und PyTorch mit MPS-Unterstützung

Hinweis: Training wurde auf einem MacBook Air M4 mit macOS Ventura oder neuer durchgeführt. Devices: CPU und MPS (Apple Silicon GPU)


Datensatz: Du kannst den Shakespeare-Datensatz von [hier](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) herunterladen.
```shell
DATAPATH="data/tinyshakespeare.txt"
curl -O {DATAPATH} https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Die Schritte im Überblick

1. **Tokenization (Der Anfang)** 
LLMs verstehen keine Wörter, sie verstehen Zahlen.
- *Konzept:* Du nimmst den Shakespeare-Text und zerlegst ihn. Im „Hello World" machen wir das auf Character-Level (Zeichen-Ebene).
- *Im Code:* Du erstellst ein Mapping: a = 1, b = 2, usw.
- *Praxis:* Das ist einfacher als die komplexe „Byte-Pair Encoding" (BPE), die ChatGPT nutzt, aber das Prinzip ist identisch.

2. **Embeddings (Die Bedeutung)** 
- *Konzept:* Jedes Token (Zahl) wird in einen Vektor (eine Liste von z.B. 64 Fließkommazahlen) umgewandelt. In diesem Vektorraum haben ähnliche Buchstaben/Wörter ähnliche mathematische Positionen.
- Im Code: Eine nn.Embedding Schicht in PyTorch.

3. **Der Transformer Block (Statt RNN)**
Hier passiert die Magie.

- *Self-Attention:* Das Modell lernt, worauf es im Satz achten muss. Wenn da steht „Er nahm den Apfel und aß...“, weiß das Modell durch Attention, dass sich „aß“ auf „Apfel“ bezieht, nicht auf „Er“.

- *Encoder/Decoder:* GPT-Modelle nutzen nur den Decoder-Teil der ursprünglichen Transformer-Architektur. Sie nehmen den bisherigen Text und sagen das nächste Token voraus.

4. **Training Loop (Das Lernen)**
- Das Modell rät das nächste Zeichen.

- Wir vergleichen es mit dem echten Shakespeare-Text.

- Wir berechnen den Fehler (Loss) und passen die Gewichte an (Backpropagation).

5. **Deployment (Interaktiver Chat)**
- Nach dem Training hast du ein Modell, das Shakespeare-ähnlichen Text generiert.

- Wir schreiben eine kleine Python-Schleife while True:, die deinen Input nimmt und die Antwort des Modells streamt.

## Installation der Abhängigkeiten
Stelle sicher, dass du die notwendigen Bibliotheken installiert hast. Hier wird Anaconda für das virtuelle Environment verwendet, du kannst aber auch pip nutzen. Die Datei `envs/llm.yml` enthält alle benötigten Pakete.
```bash
conda env create -f envs/llm.yml
conda activate llm
```

## Der Transformer: Theoretische Grundlagen

Der Transformer-Block löste die RNNs (Recurrent Neural Networks) ab, weil er parallelisiert werden kann und ein viel besseres Langzeitgedächtnis besitzt. Der Schlüssel dazu ist die Self-Attention.

### 1. **Das Eingangs-Embedding und die Position**

Ein Token (eine Zahl) wird in einen Vektor $x$ der Dimension $d_{\text{model}}$ (bei uns $n_{\text{embd}}$) umgewandelt.

$$X_{\text{Input}} = \text{TokenEmbed}(T) + \text{PositionalEmbed}(P)$$
- Token Embeddings: Enthält die semantische Bedeutung des Tokens.
- Positional Embeddings: Der Transformer hat kein inhärentes Wissen über die Reihenfolge. Daher fügen wir den Positions-Vektor hinzu, damit das Modell weiß, ob ein Wort am Anfang oder am Ende des Satzes steht.

### 2. **Die Kernidee: Query, Key und Value (Q, K, V)**

Jedes Eingangs-Embedding $x_i$ wird in drei separate Vektoren projiziert, indem es mit drei verschiedenen, lernbaren Gewichtsmatrizen $W_Q$, $W_K$ und $W_V$ multipliziert wird.

$$Q = X W_Q \quad \quad K = X W_K \quad \quad V = X W_V$$

- $Q$ (Query): Der Vektor des aktuellen Tokens, das Aufmerksamkeit sucht. ("Was suche ich?")
- $K$ (Key): Die Vektoren aller Tokens, die potenziell relevante Informationen liefern könnten. ("Was habe ich zu bieten?")
- $V$ (Value): Die tatsächlichen Informationen, die bei Relevanz weitergegeben werden. ("Das ist meine Information.")

### 3. **Die Scaled Dot-Product Attention**

Dies ist die eigentliche Formel, die wir im Code als wei = q @ k.transpose(-2, -1) * C**-0.5 gesehen haben.$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**A. Ähnlichkeit berechnen ($QK^T$)**

Die Multiplikation von $Q$ mit der transponierten Matrix $K$ ($QK^T$) führt zu einer Matrix von Attention Scores (bei uns wei).Jeder Eintrag $(i, j)$ in dieser Matrix ist ein Dot-Product (Skalarprodukt) zwischen der Query $q_i$ und dem Key $k_j$.Das Ergebnis ist ein Maß für die mathematische Ähnlichkeit (Relevanz) zwischen Token $i$ und Token $j$.

**B. Skalierung ($\frac{1}{\sqrt{d_k}}$)**

Die Division durch die Quadratwurzel der Key-Dimension $\sqrt{d_k}$ (bei uns C**-0.5) ist entscheidend:Bei großen Vektordimensionen $d_k$ werden die Dot-Products sehr groß, was dazu führen kann, dass die Softmax-Funktion in Regionen mit extrem kleinen Gradienten gerät.Die Skalierung hält die Scores in einem stabilen Bereich.

**C. Maskierung ($+M$)**

Die Maske $M$ (bei uns tril oder masked_fill) sorgt dafür, dass ein Token bei der Vorhersage keine Informationen von nachfolgenden Tokens erhalten kann.Für alle zukünftigen Tokens $j > i$ wird der Attention Score auf $-\infty$ gesetzt.Nach der Softmax-Funktion werden diese Werte zu $0$, was verhindert, dass das Modell "spickt".Dies definiert einen Decoder-Only Transformer (wie GPT).

**D. Softmax**

Die Softmax-Funktion wandelt die Scores in Wahrscheinlichkeiten um, sodass ihre Summe $1$ ergibt. Diese Wahrscheinlichkeiten (wei) bestimmen, wie viel jedes frühere Token zur Ausgabe des aktuellen Tokens beiträgt.

**E. Value-Aggregation ($\dots V$)**

Schließlich werden diese gewichteten Wahrscheinlichkeiten mit der Value-Matrix $V$ multipliziert.Die Ausgabe der Attention ist eine gewichtete Summe der Value-Vektoren, wobei die Gewichte die Softmax-Wahrscheinlichkeiten sind.Wenn Token $j$ für Token $i$ als sehr relevant erachtet wird, fließt der Value von $j$ stark in die neue Repräsentation von $i$ ein.

### 4. **Multi-Head Attention**

Im Code haben wir MultiHeadAttention mit $n_{\text{head}}=6$ implementiert.Anstatt nur eine große Attention-Berechnung durchzuführen, werden mehrere, parallele Attention-Heads verwendet.Jeder Head hat seine eigenen, unabhängigen $W_Q, W_K, W_V$ Matrizen und lernt, auf einen anderen Aspekt des Satzes zu achten (z.B. ein Head auf Grammatik, ein anderer auf Namen, ein dritter auf Zeitformen).Die Ausgaben aller Heads werden am Ende konkateniert und durch eine finale lineare Schicht (Projection self.proj) zusammengeführt.

### 5. **Der Rest des Blocks**

Nach der Multi-Head Attention folgen zwei weitere kritische Komponenten:
- Residual Connection (Restnetzwerk): $(x + Attention(x))$. Das Ergebnis der Attention wird zur ursprünglichen Eingabe addiert. Dies ermöglicht es tiefen Netzen, effizienter zu trainieren, indem der Gradientenfluss erleichtert wird.
- Layer Normalization: Normalisiert die Vektoren über die Features hinweg (anstatt über den Batch, wie bei Batch Normalization). Dies stabilisiert das Training.
- Feed Forward Network (FFN): Ein einfaches, kleines, voll verbundenes neuronales Netz. Es ist dafür da, dass jedes Token "alleine denken" kann. Es enthält meist eine Expansion ($4 \times d_{\text{model}}$) und eine Nicht-Linearität (ReLU).

Die Abfolge ist daher:

$$\begin{align*}
\text{Output} &= \text{LayerNorm}(\text{Input} + \text{Attention}(\text{Input})) \\
\text{Final} &= \text{LayerNorm}(\text{Output} + \text{FFN}(\text{Output}))
\end{align*}$$

Diese Kombination von Attention, Layer Norm und Residual Connections bildet den Transformer Block, den wir sechsmal gestapelt haben (n_layer=6).

# Nächste Schritte zur Verbesserung von nanoChat

## Fünf Schlüsselaspekte

1. ⚙️ Die Architektur-Optimierung (Tokenization)
Ihr aktuelles Modell nutzt Character-Level Tokenization. Das ist ineffizient und limitiert die Modellleistung drastisch, weil das Modell die Semantik von Wörtern nicht erkennt.

✅ Aktion: Wechsel zu BPE (Byte-Pair Encoding)
- Problem: Das Vokabular ist zu klein (nur ca. 65 Zeichen). Das Modell muss jedes Wort aus diesen 65 Zeichen zusammensetzen. Es verschwendet Attention-Kapazität für die Rechtschreibung.

- Lösung: Implementieren Sie einen Subword Tokenizer wie BPE.

  - Werkzeug: Nutzen Sie tiktoken (das ist der OpenAI-Tokenizer) oder die tokenizers-Bibliothek von Hugging Face.

  - Ergebnis: Das Vokabular wächst (z.B. auf 5.000 oder 10.000 Tokens), aber häufige Wörter wie "the", "and" oder "ing" werden zu einem einzigen Token. Dies erhöht die Informationsdichte der Eingabe signifikant.

- Effekt: Sie verbessern die Flüssigkeit (Fluency) und die Kohärenz des generierten Textes dramatisch.



## Training von Transformern

Das Training mit Batches und Gradienten in den $Q, K, V$ Matrizen ist im Grunde die Anwendung der Kettenregel der Differentialrechnung auf die gesamte, sehr lange Formel des Transformer-Blocks.

**I. Die Rolle der Batches (Parallelisierung)**

Im Gegensatz zum Bigram-Modell, das nur ein Token zurzeit betrachtet, erlauben Batches dem Transformer, extrem effizient zu arbeiten:
1. Parallelisierung: Im Training führen wir nicht nur eine Sequenz (z.B. einen Satz) durch den Transformer, sondern $B$ Sequenzen parallel (batch_size = 64).
2. Effizienz: Die gesamte Berechnung der $Q, K, V$ Matrizen für alle Tokens in allen $B$ Sequenzen erfolgt auf der GPU (deinem M4 Chip) in einem einzigen, massiven Matrixmultiplikationsschritt.
3. Stabilität: Das Training auf Batches führt zu einer stabileren Schätzung des Gesamtgradienten und hilft dem Optimierer, schneller und zuverlässiger zu konvergieren.

Der Input in den $Q, K, V$ Schichten ist somit nicht nur das Embedding eines Tokens, sondern ein großer Tensor der Dimension $(B, T, d_{\text{model}})$, der durch die $W_Q, W_K, W_V$ Matrizen läuft.

II. Die lernbaren Parameter

Die $Q, K, V$ Vektoren selbst sind keine lernbaren Parameter. Sie sind das Ergebnis der Multiplikation.Die eigentlichen lernbaren Parameter sind die drei Gewichtsmatrizen (Weight Matrices):
- $W_Q$ (für die Query-Projektion)
- $W_K$ (für die Key-Projektion)
- $W_V$ (für die Value-Projektion)
Diese Matrizen werden zu Beginn zufällig initialisiert und enthalten alle Informationen, die das Attention-Modul lernt. Die Aufgabe des Trainings ist es, diese Matrizen so anzupassen, dass sie die optimalen $Q, K, V$ Vektoren produzieren.

**III. Der Gradientenfluss (Backpropagation)**

Der Trainingsprozess folgt immer diesen Schritten: Forward Pass $\to$ Loss $\to$ Backward Pass $\to$ Update.

1. **Forward Pass & Loss**

    1. Die Eingabedaten ($X$) laufen durch die $W_Q, W_K, W_V$ Matrizen, erzeugen die Attention-Gewichte, aggregieren die Values und erzeugen am Ende des Blocks einen Output.
    2. Am Ende des gesamten Modells wird dieser Output mit den tatsächlichen Targets verglichen, um den Verlust (Loss $L$, bei uns Cross-Entropy) zu berechnen.

2. **Backward Pass: Die Kettenregel**

Der Loss $L$ ist der Ausgangspunkt. Die Backpropagation berechnet die Ableitung (den Gradienten) des Loss in Bezug auf alle Parameter. Das heißt, wir berechnen:

$$\frac{\partial L}{\partial W_Q}, \quad \frac{\partial L}{\partial W_K}, \quad \frac{\partial L}{\partial W_V}$$

Wie das Signal durch die Attention fließt:

Das Fehlersignal muss rückwärts durch die gesamte Attention-Formel laufen:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

- Der Gradient geht rückwärts durch $V$: Der Gradient des Loss kommt in den $V$-Pfad. Er muss berechnet werden in Bezug auf $W_V$, um zu lernen, welche Informationen (Values) gewichtet werden sollen.
- Der Gradient geht rückwärts durch Softmax: Der Fehler muss durch die nicht-lineare Softmax-Funktion fließen, was die Berechnung kompliziert macht, aber essenziell ist.
- Der Gradient teilt sich in $Q$ und $K$: Die komplexeste Stelle ist die Multiplikation $QK^T$. Der Fehler, der über die Attention Scores hereinkommt, muss auf die $Q$- und $K$-Matrizen aufgeteilt werden (Kettenregel).
    - Der Gradient $\frac{\partial L}{\partial W_Q}$ sagt dem Modell: "Wenn ich dieses Token als Query hatte, war das Ergebnis falsch. Ich muss die Query-Repräsentation (die Wichtigkeit des Suchens) anpassen."
    - Der Gradient $\frac{\partial L}{\partial W_K}$ sagt dem Modell: "Das Key-Vektor-Angebot dieses Tokens führte zu einem Fehler. Ich muss die Key-Repräsentation (das, was das Token anbietet) anpassen."

3. **Update (Optimierer)**

Nachdem die Gradienten $\frac{\partial L}{\partial W}$ für alle Gewichtsmatrizen (nicht nur $W_Q, W_K, W_V$, sondern auch die Feed-Forward-Layer) berechnet wurden, nutzt der Optimierer (z.B. AdamW) diese, um die Parameter zu aktualisieren:

$$W_{\text{neu}} = W_{\text{alt}} - \eta \cdot \nabla L$$

Wobei $\eta$ die Lernrate (learning_rate) ist. Die $W_Q, W_K, W_V$ Matrizen werden schrittweise in die Richtung des stärksten Abstiegs des Fehlers (entgegengesetzt zum Gradienten) verschoben. 
- Mit jedem Schritt lernt das Modell, welche $Q$ am besten zu welchen $K$ passen, um am Ende die korrekten $V$ zu aggregieren.

**Zusammenfassend**: Das Training eines Transformers ist ein hochparalleler Prozess, bei dem der vom Loss berechnete Fehler über die Kettenregel durch alle Schichten und jede einzelne Matrixmultiplikation zurückgereicht wird, um die Gewichte der $W_Q, W_K, W_V$ Matrizen anzupassen.