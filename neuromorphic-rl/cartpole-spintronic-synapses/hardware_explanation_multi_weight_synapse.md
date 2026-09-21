# Hardware Background for Multi-Weight Molecular Spintronic Synapses

This note explains the hardware part of the project in simple words, step by step.  
It is written as a presentation-preparation document: first understand the device, then understand how it represents neural-network weights, and then connect it to the DQN CartPole project.

---

## 1. Main idea

In a normal neural network, each synapse has a numerical weight:

\[
w
\]

In software, this weight is stored as a floating-point number in memory.

In neuromorphic/in-memory hardware, the idea is different:

> A physical device acts as the synapse, and its electrical conductance represents the neural-network weight.

So the mapping is:

| Neural-network concept | Hardware concept |
|---|---|
| Synapse | Physical device / crosspoint |
| Weight | Conductance |
| Learning | Changing conductance |
| Input activation | Applied voltage |
| Weighted output | Current |

The key idea is:

\[
\text{weight} \approx \text{conductance}
\]

This is useful because computation and memory are located in the same place. The device both stores the weight and participates in the calculation.

---

## 2. Basic electrical formulas

Before understanding the hardware, we need four electrical quantities.

### Voltage \(V\)

Voltage is the electrical “push” that drives current through a device.

Unit:

\[
V
\]

### Current \(I\)

Current is the flow of electric charge.

Unit:

\[
A
\]

### Resistance \(R\)

Resistance tells how strongly the device blocks current.

Unit:

\[
\Omega
\]

High resistance means current flows with difficulty.

### Conductance \(G\)

Conductance tells how easily current flows.

Unit:

\[
S
\]

Conductance is the inverse of resistance:

\[
G = \frac{1}{R}
\]

and:

\[
R = \frac{1}{G}
\]

So:

| Resistance | Conductance |
|---|---|
| high \(R\) | low \(G\) |
| low \(R\) | high \(G\) |

### Ohm’s law

The basic relation is:

\[
V = IR
\]

From this:

\[
I = \frac{V}{R}
\]

Because:

\[
G = \frac{1}{R}
\]

we can write:

\[
I = GV
\]

This is the most important formula for neuromorphic crossbars.

It means:

\[
\text{current} = \text{conductance} \times \text{voltage}
\]

In neural-network language:

\[
\text{output contribution} = \text{weight} \times \text{input}
\]

So the hardware relation:

\[
I = GV
\]

matches the neural-network multiplication:

\[
\text{weighted input} = w x
\]

where:

- input \(x\) is represented by voltage \(V\),
- weight \(w\) is represented by conductance \(G\),
- output contribution is represented by current \(I\).

---

## 3. Simple numerical example: reading conductance

Suppose we apply a small read voltage:

\[
V_{read} = 0.1V
\]

and measure current:

\[
I = 20\mu A
\]

Then conductance is:

\[
G = \frac{I}{V}
\]

\[
G = \frac{20\mu A}{0.1V}
\]

\[
G = 200\mu S
\]

Resistance is:

\[
R = \frac{1}{G}
\]

\[
R = \frac{1}{200\mu S}
\]

\[
R = 5k\Omega
\]

So by applying a small voltage and measuring current, we can calculate the conductance.

---

## 4. What is the hardware device?

The hardware considered in the project is based on molecular spintronic resistive-switching devices.

The structure described in the article is:

\[
\text{LSMO} / \text{Alq3} / \text{AlOx} / \text{Co}
\]

A simplified structure is:

```text
Top electrode:       Co
                     │
Middle layers:       Alq3 + AlOx
                     │
Bottom electrode:    LSMO
```

Current flows between the top and bottom electrodes through the molecular/tunnel layers.

This device is useful because it combines two effects:

1. **Resistive switching**
2. **Magnetoresistance**

These two effects together allow the device to work as a multi-weight artificial synapse.

---

## 5. Resistive switching

### Simple definition

Resistive switching means:

> The resistance of the device can be changed by applying voltage pulses.

Since:

\[
G = \frac{1}{R}
\]

changing resistance means changing conductance.

So:

\[
\text{voltage pulse} \rightarrow R \text{ changes} \rightarrow G \text{ changes} \rightarrow w \text{ changes}
\]

### Why this matters

If conductance represents the synaptic weight, then changing conductance means changing the neural-network weight.

So resistive switching is the mechanism that allows learning.

### High-resistance and low-resistance states

The device can be in different resistance states:

| State | Resistance | Conductance | Synaptic meaning |
|---|---|---|---|
| High-resistance state | high \(R\) | low \(G\) | weak synapse |
| Low-resistance state | low \(R\) | high \(G\) | strong synapse |

Some devices are not only binary. They can have many intermediate conductance states, which is useful for analog neural-network weights.

---

## 6. Voltage pulses: read versus write

There are two important voltage operations.

### 6.1 Read voltage

A read voltage is small.

Purpose:

> Measure conductance without changing the device.

Steps:

1. Apply a small voltage \(V_{read}\).
2. Measure current \(I\).
3. Compute conductance:

\[
G = \frac{I}{V_{read}}
\]

Example:

```text
Small voltage → measure current → calculate conductance
```

### 6.2 Write/update voltage

A write voltage is a larger pulse.

Purpose:

> Change the resistive state of the device.

Example pulse:

```text
0 V → +2 V for a short time → 0 V
```

or:

```text
0 V → -2 V for a short time → 0 V
```

The exact pulse polarity and amplitude depend on the device.

A safe presentation sentence is:

> “A small voltage is used for reading, while larger voltage pulses are used for writing or updating the conductance.”

---

## 7. Where is voltage applied?

A crosspoint device is like a small sandwich:

```text
Top electrode
     │
Device material
     │
Bottom electrode
```

Voltage is applied between the top electrode and the selected bottom electrode.

For the molecular spintronic device:

```text
Top electrode:       Co
Bottom electrode:    LSMO
```

So we apply voltage between Co and the selected LSMO electrode.

In the multi-weight synapse, there are several bottom electrodes and one shared top electrode:

```text
              Common top Co electrode
        ─────────────────────────────────
              │          │          │
           device 0   device 1   bias device
              │          │          │
           LSMO 0     LSMO 1     LSMO bias
```

To program device 0, apply voltage between:

```text
Co top electrode ↔ LSMO 0 bottom electrode
```

To program device 1, apply voltage between:

```text
Co top electrode ↔ LSMO 1 bottom electrode
```

To program the bias device, apply voltage between:

```text
Co top electrode ↔ LSMO bias bottom electrode
```

---

## 8. SET / potentiation and RESET / depression

Voltage pulses can move the device conductance up or down.

### SET / potentiation

SET or potentiation means conductance increases:

\[
G \uparrow
\]

Because:

\[
G = \frac{1}{R}
\]

this means resistance decreases:

\[
R \downarrow
\]

Neural-network meaning:

> The synapse becomes stronger.

Example:

\[
G: 200\mu S \rightarrow 250\mu S
\]

### RESET / depression

RESET or depression means conductance decreases:

\[
G \downarrow
\]

and resistance increases:

\[
R \uparrow
\]

Neural-network meaning:

> The synapse becomes weaker.

Example:

\[
G: 200\mu S \rightarrow 150\mu S
\]

### Important clarification

We should not say:

> “Reducing voltage always increases conductance.”

That is not generally correct.

Better:

> “The direction of conductance change depends on the pulse polarity, pulse amplitude, pulse duration, selected electrode, and device structure.”

---

## 9. Example of pulse-based programming

A simple example:

1. Start with:

\[
G = 200\mu S
\]

2. Apply a potentiation pulse.

3. Read again:

\[
G = 250\mu S
\]

So conductance increased:

\[
200\mu S \rightarrow 250\mu S
\]

This means the weight increased.

Another example:

1. Start with:

\[
G = 200\mu S
\]

2. Apply a depression pulse.

3. Read again:

\[
G = 150\mu S
\]

So conductance decreased:

\[
200\mu S \rightarrow 150\mu S
\]

This means the weight decreased.

---

## 10. Crossbar computation

Many devices can be arranged in a crossbar array.

Simplified crossbar:

```text
Input voltages on rows
        V1   V2   V3
        │    │    │
        G11  G12  G13  → output current column 1
        G21  G22  G23  → output current column 2
        G31  G32  G33  → output current column 3
```

Each crosspoint has a conductance.

For one crosspoint:

\[
I_{ij} = G_{ij}V_i
\]

The currents in the same column add naturally by Kirchhoff’s current law:

\[
I_j = \sum_i G_{ij}V_i
\]

This is exactly matrix-vector multiplication:

\[
\mathbf{I} = \mathbf{G}\mathbf{V}
\]

In neural-network language:

\[
\text{output} = \text{weight matrix} \times \text{input vector}
\]

So the hardware naturally performs the expensive matrix multiplication.

---

## 11. \(V_{write}/2\) protocol

In a crossbar, many devices share wires. If we apply a full write voltage carelessly, we may accidentally update unselected devices.

The \(V_{write}/2\) protocol helps solve this.

Basic idea:

- selected row gets \(+V_{write}/2\),
- selected column gets \(-V_{write}/2\),
- selected device feels the full voltage:

\[
V_{write}
\]

Example:

```text
selected row = +1 V
selected column = -1 V
```

The selected crosspoint sees:

\[
(+1V) - (-1V) = 2V
\]

So:

\[
V_{write} = 2V
\]

Unselected devices see only half voltage or zero voltage, so they should not switch.

Presentation sentence:

> “The \(V_{write}/2\) method lets us update one selected synapse while avoiding unwanted updates in neighboring synapses.”

---

## 12. Magnetoresistance

### Simple definition

Magnetoresistance means:

> The resistance of the device depends on the magnetic configuration.

The device has magnetic electrodes. Their magnetizations can be aligned in different ways.

There are two important states:

1. Parallel state \(P\)
2. Antiparallel state \(AP\)

---

## 13. Parallel state \(P\)

In the parallel state, the magnetic directions point the same way.

Example:

```text
Reference layer:  →
Free layer:       →
```

The conductance in this state is:

\[
G_P
\]

So:

\[
G_P = \text{conductance when the magnetic layers are parallel}
\]

---

## 14. Antiparallel state \(AP\)

In the antiparallel state, the magnetic directions point in opposite directions.

Example:

```text
Reference layer:  →
Free layer:       ←
```

The conductance in this state is:

\[
G_{AP}
\]

So:

\[
G_{AP} = \text{conductance when the magnetic layers are antiparallel}
\]

Because of magnetoresistance:

\[
R_P \neq R_{AP}
\]

and therefore:

\[
G_P \neq G_{AP}
\]

Important safe sentence:

> “The P and AP states have different resistance, and therefore different conductance.”

Do not necessarily say one is always larger unless you are referring to a specific measured device.

---

## 15. How do we switch between \(P\) and \(AP\)?

To switch between \(P\) and \(AP\), we must reverse the magnetization of the free magnetic layer.

### Conceptual picture

Initial \(P\) state:

```text
Reference layer:  →
Free layer:       →
```

After switching:

```text
Reference layer:  →
Free layer:       ←
```

Now the device is in \(AP\).

### Experimental characterization

In many experiments, researchers use an external magnetic field to characterize P and AP states.

A magnetic field can force magnetic layers to switch, allowing the measurement of different resistance/conductance states.

### Practical integrated hardware: spin-orbit torque

For real integrated hardware, using a large external magnetic field is inconvenient.

A practical electrical method is **spin-orbit torque**, or **SOT**.

SOT switching works like this:

```text
current pulse through spin-orbit layer
        ↓
spin accumulation / spin current
        ↓
torque on free magnetic layer
        ↓
free layer flips
        ↓
P changes to AP, or AP changes to P
```

Simplified picture:

```text
        Free magnetic layer
        ↑ magnetization can flip
   ----------------------------
        Spin-orbit layer
   ----------------------------
 current pulse flows sideways →
```

The important distinction is:

| Mechanism | What it changes |
|---|---|
| Voltage pulse / resistive switching | changes conductance level |
| SOT pulse | changes magnetic configuration \(P\) or \(AP\) |

Presentation sentence:

> “Voltage pulses tune the conductance value, while spin-orbit torque pulses select whether a crosspoint is read in the parallel or antiparallel magnetic state.”

---

## 16. Why combine resistive switching and magnetoresistance?

Each effect has a different role.

| Effect | Controlled by | What changes? | Role |
|---|---|---|---|
| Resistive switching | voltage pulses | resistive/conductance level | stores/tunes the weight value |
| Magnetoresistance | magnetic state | P or AP conductance | selects which effective weight is recalled |

Simple explanation:

> Resistive switching stores the value. Magnetoresistance selects which version of the value we read.

This combination is the reason the hardware can support multi-weight synapses.

---

## 17. Multi-weight synapse

A normal synapse gives one weight.

A multi-weight synapse can produce different effective weights depending on the selected task.

For two tasks, one synapse contains:

1. task 0 positive crosspoint,
2. task 1 positive crosspoint,
3. shared bias/reference crosspoint.

Simplified structure:

```text
              Common top electrode
        ─────────────────────────────
            │          │          │
        task 0      task 1      bias
        device      device      device
            │          │          │
        bottom 0   bottom 1   bottom bias
```

The task index decides which crosspoint contributes as \(G_{AP}\), and which contributes as \(G_P\).

---

## 18. Effective weight for task 0

For task 0:

\[
w_0 = \alpha \left(G_{AP,0} + G_{P,1} - G_{bias}\right)
\]

Meaning:

| Component | Role |
|---|---|
| \(G_{AP,0}\) | task 0 crosspoint is active, read in AP state |
| \(G_{P,1}\) | task 1 crosspoint is inactive, read in P state |
| \(-G_{bias}\) | bias/reference conductance is subtracted |
| \(\alpha\) | scaling factor |

---

## 19. Effective weight for task 1

For task 1:

\[
w_1 = \alpha \left(G_{P,0} + G_{AP,1} - G_{bias}\right)
\]

Meaning:

| Component | Role |
|---|---|
| \(G_{P,0}\) | task 0 crosspoint is inactive, read in P state |
| \(G_{AP,1}\) | task 1 crosspoint is active, read in AP state |
| \(-G_{bias}\) | bias/reference conductance is subtracted |
| \(\alpha\) | scaling factor |

So:

| Selected task | Crosspoint 0 | Crosspoint 1 | Bias | Weight |
|---|---|---|---|---|
| Task 0 | \(G_{AP,0}\) | \(G_{P,1}\) | \(-G_{bias}\) | \(w_0\) |
| Task 1 | \(G_{P,0}\) | \(G_{AP,1}\) | \(-G_{bias}\) | \(w_1\) |

---

## 20. Why subtract \(G_{bias}\)?

Conductance is normally positive.

But neural-network weights can be:

- positive,
- negative,
- close to zero.

The bias/reference conductance allows signed weights.

General form:

\[
w = G_{positive} - G_{bias}
\]

If:

\[
G_{positive} > G_{bias}
\]

then:

\[
w > 0
\]

If:

\[
G_{positive} < G_{bias}
\]

then:

\[
w < 0
\]

If:

\[
G_{positive} \approx G_{bias}
\]

then:

\[
w \approx 0
\]

Important note:

> This hardware bias/reference conductance is not the same as the neural-network bias parameter. It is a physical subtraction term used to create signed effective weights.

---

## 21. What does \(\alpha\) do?

The scaling factor \(\alpha\) converts conductance values into the numerical range needed by the neural network.

\[
w = \alpha(\text{conductance combination})
\]

If conductance values are too large compared with neural-network weights, \(\alpha\) can make them smaller.

If conductance values are too small, \(\alpha\) can scale them up.

Presentation sentence:

> “\(\alpha\) maps the physical conductance scale to the numerical weight scale used by the neural network.”

---

## 22. How are \(G_P\), \(G_{AP}\), and \(G_{bias}\) calculated in the simulation?

In real hardware, conductance is measured using:

\[
G = \frac{I}{V}
\]

But in the simulation, conductance is modeled mathematically from an internal state \(x\).

The formulas used in the project are:

\[
G_P(x) = g_p\log_{10}(x)
\]

\[
G_{AP}(x) = g_{ap}\log_{10}(x)
\]

\[
G_{bias}(x) = g_{bias}\log_{10}(x)
\]

where:

| Symbol | Meaning |
|---|---|
| \(x\) | internal crosspoint state |
| \(g_p\) | scaling factor for parallel conductance |
| \(g_{ap}\) | scaling factor for antiparallel conductance |
| \(g_{bias}\) | scaling factor for bias conductance |

So the full chain is:

```text
voltage pulses → change internal state x → calculate conductance G(x) → calculate weight w
```

---

## 23. Why use \(\log_{10}(x)\)?

Real resistive-switching devices often do not change conductance perfectly linearly.

A logarithmic model captures a nonlinear response:

\[
G(x) = g\log_{10}(x)
\]

Example:

| \(x\) | \(\log_{10}(x)\) |
|---:|---:|
| 1 | 0 |
| 10 | 1 |
| 100 | 2 |
| 1000 | 3 |

So if \(x\) increases a lot, conductance increases more slowly.

Presentation sentence:

> “The logarithmic function is used to model the nonlinear relationship between the internal resistive state and conductance.”

---

## 24. Numerical example: conductance calculation

Assume:

\[
x = 100
\]

Then:

\[
\log_{10}(100)=2
\]

Let:

\[
g_p = 1
\]

\[
g_{ap} = 3
\]

\[
g_{bias} = 2
\]

Then:

\[
G_P = g_p\log_{10}(x) = 1 \times 2 = 2
\]

\[
G_{AP} = g_{ap}\log_{10}(x) = 3 \times 2 = 6
\]

\[
G_{bias} = g_{bias}\log_{10}(x) = 2 \times 2 = 4
\]

If the simple weight formula is:

\[
w = G_{AP} + G_P - G_{bias}
\]

then:

\[
w = 6 + 2 - 4 = 4
\]

So the effective weight is:

\[
w = 4
\]

---

## 25. Numerical example: task-specific weights

Assume:

\[
\alpha = 0.5
\]

and:

\[
G_{AP,0}=8
\]

\[
G_{P,1}=3
\]

\[
G_{P,0}=4
\]

\[
G_{AP,1}=10
\]

\[
G_{bias}=6
\]

### Task 0

\[
w_0 = \alpha(G_{AP,0}+G_{P,1}-G_{bias})
\]

\[
w_0 = 0.5(8+3-6)
\]

\[
w_0 = 0.5(5)
\]

\[
w_0 = 2.5
\]

### Task 1

\[
w_1 = \alpha(G_{P,0}+G_{AP,1}-G_{bias})
\]

\[
w_1 = 0.5(4+10-6)
\]

\[
w_1 = 0.5(8)
\]

\[
w_1 = 4
\]

So the same physical synapse can produce two different effective weights:

\[
w_0 = 2.5
\]

\[
w_1 = 4
\]

This is the multi-weight idea.

---

## 26. How learning updates the synapse

In normal deep learning, a weight is updated using something like:

\[
w \leftarrow w - \eta \nabla w
\]

where:

- \(\eta\) is the learning rate,
- \(\nabla w\) is the gradient.

But this is a floating-point software update.

Physical conductance devices are better suited to pulse-based updates.

So in this project, backpropagation is used only to find the sign of the gradient:

- positive gradient,
- negative gradient,
- zero gradient.

Then the synaptic controller applies a hardware-inspired update.

| Gradient sign | Desired effect | Synaptic update | Weight effect |
|---|---|---|---|
| positive gradient | weight should decrease | increase bias crosspoint | \(w\) decreases |
| negative gradient | weight should increase | increase active task positive crosspoint | \(w\) increases |
| zero gradient | no change | no update | \(w\) unchanged |

This is a sign-based or Manhattan-style update.

---

## 27. Why positive gradient means weight decreases

Gradient descent updates weights in the opposite direction of the gradient:

\[
w \leftarrow w - \eta \nabla w
\]

If:

\[
\nabla w > 0
\]

then:

\[
w
\]

should decrease.

In this hardware-inspired model:

\[
w = G_{positive} - G_{bias}
\]

To decrease \(w\), we can increase \(G_{bias}\):

\[
G_{bias} \uparrow \Rightarrow w \downarrow
\]

Example:

Before:

\[
w = 8 + 3 - 6 = 5
\]

Increase bias:

\[
G_{bias}: 6 \rightarrow 7
\]

After:

\[
w = 8 + 3 - 7 = 4
\]

So the weight decreased.

---

## 28. Why negative gradient means weight increases

If:

\[
\nabla w < 0
\]

gradient descent increases the weight because:

\[
w \leftarrow w - \eta(\text{negative})
\]

means:

\[
w \uparrow
\]

In the hardware-inspired model, to increase \(w\), increase the active positive conductance.

Example:

Before:

\[
w = 8 + 3 - 6 = 5
\]

Increase active conductance:

\[
G_{AP,0}: 8 \rightarrow 9
\]

After:

\[
w = 9 + 3 - 6 = 6
\]

So the weight increased.

---

## 29. Real hardware versus simulation

This distinction is important in a presentation.

### Real hardware

In real hardware:

1. Apply read voltage.
2. Measure current.
3. Compute conductance:

\[
G = \frac{I}{V}
\]

4. Apply write pulses to modify conductance.
5. Use magnetic state \(P\) or \(AP\) to select the conductance contribution.

### Simulation

In your simulation:

1. Each crosspoint has an internal state \(x\).
2. Conductance is modeled as:

\[
G(x)=g\log_{10}(x)
\]

3. The effective weight is calculated from conductances.
4. This weight is loaded into the PyTorch DQN.
5. Backpropagation calculates gradients.
6. The sign of the gradient updates the internal crosspoint state.

Presentation sentence:

> “The simulation does not directly measure a physical device. Instead, it uses a mathematical conductance model inspired by the real device behavior.”

---

## 30. How this connects to your DQN project

Your DQN project uses two CartPole environments:

1. standard CartPole,
2. modified CartPole.

Both tasks use the same neural-network architecture.

But the effective weights are task-dependent.

The synaptic controller does this:

```text
task index 0 → load weights w0
task index 1 → load weights w1
```

where:

\[
w_0 = \alpha(G_{AP,0}+G_{P,1}-G_{bias})
\]

\[
w_1 = \alpha(G_{P,0}+G_{AP,1}-G_{bias})
\]

So the same DQN can behave differently depending on which task is active.

---

## 31. Suggested presentation flow

You can explain your work in this order.

### Slide 1: Motivation

Modern AI needs many weight transfers between memory and processor. This is slow and energy expensive.

### Slide 2: In-memory/neuromorphic idea

Use physical devices as synapses. Store weights directly as conductance.

### Slide 3: Electrical basics

Show:

\[
G=\frac{1}{R}
\]

\[
I=GV
\]

Explain that voltage is input, conductance is weight, current is weighted output.

### Slide 4: Resistive switching

Voltage pulses change the resistance/conductance of the device.

### Slide 5: Magnetoresistance

Magnetic configuration changes conductance:

\[
G_P \neq G_{AP}
\]

### Slide 6: Switching P/AP

Explain that \(P\) and \(AP\) are selected by switching the free magnetic layer. Mention that SOT can electrically switch the magnetic state.

### Slide 7: Multi-weight synapse

Show the three-crosspoint structure:

```text
task 0 crosspoint + task 1 crosspoint - bias crosspoint
```

### Slide 8: Weight equations

Show:

\[
w_0 = \alpha(G_{AP,0}+G_{P,1}-G_{bias})
\]

\[
w_1 = \alpha(G_{P,0}+G_{AP,1}-G_{bias})
\]

### Slide 9: Conductance model

Show:

\[
G_P(x)=g_p\log_{10}(x)
\]

\[
G_{AP}(x)=g_{ap}\log_{10}(x)
\]

\[
G_{bias}(x)=g_{bias}\log_{10}(x)
\]

### Slide 10: Learning rule

Show:

| Gradient sign | Update | Effect |
|---|---|---|
| positive | increase bias | weight decreases |
| negative | increase active positive conductance | weight increases |

### Slide 11: Move to DQN

Now explain CartPole, DQN, replay memory, Bellman targets, and results.

---

## 32. Short presentation script

You can say:

> “The hardware idea in this project is to represent neural-network weights using conductance. Conductance is the inverse of resistance, and the basic electrical relation is \(I=GV\). This means that if the input is applied as a voltage and the synaptic weight is stored as conductance, the output current naturally represents multiplication between input and weight.
>
> The device used here is a molecular spintronic resistive-switching device. It combines two important effects. The first is resistive switching, where voltage pulses change the resistance and therefore the conductance of the device. This allows the synaptic weight to be programmed. The second is magnetoresistance, where the conductance depends on whether the magnetic layers are in a parallel or antiparallel state. These two states are represented by \(G_P\) and \(G_{AP}\).
>
> To switch between the parallel and antiparallel states, the magnetization of the free layer must be reversed. Experimentally this can be characterized using a magnetic field, but for integrated hardware this switching can be done electrically using spin-orbit torque. SOT uses a current pulse to create a spin torque that flips the free magnetic layer.
>
> A multi-weight synapse is built from several crosspoints. In my two-task case, each synapse contains one positive crosspoint for task 0, one positive crosspoint for task 1, and one shared bias crosspoint. For task 0, the effective weight is \(w_0=\alpha(G_{AP,0}+G_{P,1}-G_{bias})\). For task 1, the effective weight is \(w_1=\alpha(G_{P,0}+G_{AP,1}-G_{bias})\). This means that the same physical synapse can generate different weights depending on the task.
>
> In the simulation, the conductances are calculated from an internal crosspoint state \(x\), using logarithmic relations such as \(G_P(x)=g_p\log_{10}(x)\). Learning is performed using the sign of the gradient. If the gradient is positive, the weight should decrease, so the bias conductance is increased. If the gradient is negative, the weight should increase, so the active task conductance is increased. This makes the learning rule more compatible with pulse-based hardware updates.”

---

## 33. Quick questions and answers

### Q1. What is conductance?

Conductance tells how easily current flows through a device.

\[
G = \frac{1}{R}
\]

### Q2. Why is conductance used as a weight?

Because:

\[
I = GV
\]

So conductance naturally multiplies the input voltage, like a neural-network weight multiplies an input.

### Q3. What is resistive switching?

It is the ability of the device to change resistance/conductance when voltage pulses are applied.

### Q4. What is magnetoresistance?

It is the change of resistance/conductance caused by magnetic configuration.

### Q5. What are \(P\) and \(AP\)?

\(P\) means magnetic layers are parallel.  
\(AP\) means magnetic layers are antiparallel.

### Q6. How do we switch \(P\) and \(AP\)?

By flipping the free magnetic layer. In practical integrated hardware, this can be done electrically using spin-orbit torque.

### Q7. Why is there a bias conductance?

To allow effective weights to be positive or negative:

\[
w = G_{positive} - G_{bias}
\]

### Q8. What is \(x\)?

\(x\) is the internal crosspoint state in the simulation. It represents the programmed resistive state.

### Q9. How are conductances calculated in the simulation?

\[
G_P(x)=g_p\log_{10}(x)
\]

\[
G_{AP}(x)=g_{ap}\log_{10}(x)
\]

\[
G_{bias}(x)=g_{bias}\log_{10}(x)
\]

### Q10. What is the main hardware contribution?

The hardware idea allows one physical synapse to produce multiple task-specific weights by combining resistive switching with magnetoresistance.

---

## 34. One-sentence summary

> The hardware stores neural-network weights as conductance, tunes them using resistive-switching voltage pulses, selects \(G_P\) or \(G_{AP}\) using magnetic-state switching such as spin-orbit torque, and combines several conductance crosspoints to create task-specific multi-weight synapses.

---

## 35. Source notes

This explanation is based on the uploaded project report and supporting articles on molecular spintronic synapses, analog memristor reinforcement learning, and neuromorphic hardware. In particular, the project report defines the two-task weight equations and logarithmic conductance model, while the molecular spintronic article motivates the use of conductance as synaptic weight and the combination of resistive switching with magnetoresistance.
