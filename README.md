# Proactive Hearing Assistants that Isolate Egocentric Conversations

---

## Videos

- [Video 1](https://youtu.be/y_nADQxiqn8)  
- [Video 2](https://youtu.be/ffh9lHBM2HU)

---

## Motivation

Hearing what you want to hear is a challenge in complex acoustic environments.  
Human hearing is remarkably adaptive, yet crowded environments make it difficult to isolate relevant voices, which is a phenomenon known as the cocktail party problem.  
For individuals with hearing loss, these overlapping conversations often lead to cognitive overload and listening fatigue, making natural communication exhausting.
Existing hearing aids and augmented audio devices are reactive: users must manually select sound sources or adjust filters through apps or spatial controls.   
The motivation of this work is to build a hearing assistant that operates proactively, automatically inferring which conversation the wearer is engaged in and isolating it in real time.

---

## Abstract
We introduce proactive hearing assistants1 that automatically identify and separate the wearer’s conversation partners, without requiring ex- plicit prompts. Our system operates on ego- centric binaural audio and uses the wearer’s self-speech as an anchor, leveraging turn-taking behavior and dialogue dynamics to infer con- versational partners and suppress others. To enable real-time, on-device operation, we pro- pose a dual-model architecture: a lightweight streaming model runs every 12.5 ms for low- latency extraction of the conversation partners, while a slower model runs less frequently to capture longer-range conversational dynamics. Results on real-world 2- and 3-speaker conver- sation test sets, collected with binaural ego- centric hardware from 11 participants total- ing 6.8 hours, show generalization in identi- fying and isolating conversational partners in multi-conversation settings. Our work marks a step toward hearing assistants that adapt proactively to conversational dynamics and engagement.
