Project Overview
This research venture is a blueprint to explore public perception and emotional discourse regarding ant topic using a large-scale dataset from Reddit. By applying Computational Social Science (CSS) paradigms, this project seeks to uncover complex social patterns in online interactions that are often obscured in manual qualitative reviews.

Theoretical Framework
This study is guided by an integrated framework rooted in Computational Social Science (CSS) and Computer-Assisted Thematic Analysis.

Sentiment & Emotion Analysis: Guided by the valence-based models of Hutto and Gilbert (2014). We utilize Transformer-based models (Devlin et al., 2019) to move beyond keyword matching, capturing the contextual emotional weight of language in social media discourse.

Thematic Analysis: We align with Braun and Clarke’s (2021) framework for thematic analysis. By utilizing Latent Dirichlet Allocation (LDA) for first-cycle data-driven clustering, the project combines computational efficiency with a second-cycle qualitative interpretation for deep analytical depth.

Research Procedure
Following the best practices established by Bruns & Burgess (2012) and Bail (2014), the project executed a three-stage protocol:

Stage 1: Data Collection & Curation
Acquisition: Connected to the Reddit API via PRAW for broad-spectrum keyword retrieval.
Noise Reduction: Implemented programmatic filtering to mitigate the limitations of simple keyword searches (Harrower, 2021).
Preprocessing: Applied a rigorous nine-step text cleaning pipeline (normalization, lemmatization, and noise removal).

Stage 2: Computational Analysis
I triangulated findings through three parallel NLP streams:
Sentiment Analysis: Quantifying affective tone using a customized VADER model.
Emotion Classification: Utilizing a state-of-the-art Transformer model (DistilRoBERTa) to identify complex emotional archetypes.
Topic Modeling: Unsupervised LDA to identify thematic content and discursive clusters.

Stage 3: Ethical De-identification
In alignment with the Association of Internet Researchers (AoIR, 2020) and Fiesler & Proferes (2018), a final ethical pass was executed. This programmatic de-identification script redacts personally identifiable information (PII) to protect user privacy and minimize harm.
Ethical Standards may vary across organisations and location.

Disclaimer & Open Collaboration

Project Status: This is a self-initiated research project developed independently to explore the intersections of data science and social phenomena, I am a psychologist who is very interested tech.

Use of AI: In the spirit of transparency, I acknowledge the use of AI tools (including Google AI Studio and Large Language Models) to assist in writing the code, and refining the documentation. This project serves as a practical application of human-AI collaboration in computational research and I hope to make it up for the water I've wasted. I have tried and failed to run AI offline mostly due to lack of GPU. I've learnt coding through youtube and hence the AI dependence (Not an excuse ik) 

Open to Feedback: As this is an independent project, I am fully open to and welcome suggestions, methodological critiques, or recommendations for further optimization. 
Whether you have ideas for improving the LDA topic modeling or suggestions for additional data sources, please feel free to:

Open an Issue in this repository.
Submit a Pull Request.
Reach out with professional feedback.

References
Braun, V., & Clarke, V. (2021). Thematic Analysis: A Practical Guide.
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis.
Lazer, D., et al. (2009). Computational Social Science. Science.
Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.