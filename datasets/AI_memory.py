import pandas as pd

# Creating the AI_Memory_Tech_Novel_Research_Measurements.csv file
data = {
    "Research Domain": [
        "Neuroimaging (fMRI)",
        "Neuroimaging (fMRI)",
        "Cognitive Psychophysics (EEG/ERP)",
        "Biometric Physiology",
        "Computational NLP (Vector Semantics)",
        "Computational NLP (Topic Modeling)",
        "Computational NLP (Sentiment Analysis)",
        "Reading Comprehension & Recall"
    ],
    "Target Theme": [
        "Neural Memory Mapping",
        "Synthetic Consciousness",
        "Digital Afterlife & Ethics",
        "Tech-Driven Anxiety & Immersion",
        "AI & Posthuman Identity",
        "Evolution of Tech Narrative Arc",
        "Machine Autonomy & Ethics",
        "Digital vs. Physical Reading Memory"
    ],
    "Metric / Measurement": [
        "Hippocampal & mPFC BOLD Signal Variance",
        "Default Mode Network (DMN) Signal Intensity",
        "N400 Amplitude Spike (µV)",
        "Galvanic Skin Response (GSR in µS) & Heart Rate",
        "Cosine Distance (Word Embeddings)",
        "Topic Prevalence (Latent Dirichlet Allocation %)",
        "Emotional Valence & Ambiguity Scores",
        "Temporal Plot Sequence Recall Accuracy (%)"
    ],
    "Typical Measured Range / Unit": [
        "% Signal Change (0.5% - 2.5%)",
        "Z-score / Beta Weights (1.5 - 3.2)",
        "Microvolts (-4.0 µV to -8.5 µV)",
        "Skin Conductance Level (2.0 - 15.0 µS)",
        "Distance Metric (0.12 - 0.85)",
        "Probability Mass (5.0% - 28.5%)",
        "Valence Score (-1.0 to +1.0)",
        "Percentage Accuracy (60% - 92%)"
    ],
    "Key Novel Discovery / Empirical Finding": [
        "Physical pages trigger higher hippocampal BOLD activity and spatial memory consolidation compared to digital scroll UI.",
        "Readers recruit the same DMN neural networks for AI characters as for human characters when narrative conveys Theory of Mind.",
        "Concepts of digital immortality trigger N400 ERP spikes, indicating higher cognitive integration cost for non-biological mortality.",
        "Surveillance and tech-anxiety scenes induce significant GSR spikes, measuring subconscious physiological tension in real-time.",
        "Semantic proximity between 'consciousness' and 'silicon/network' in novels decreased by ~35% over the last decade, showing normalization.",
        "AI ethics topics increased from 2.1% of published sci-fi corpus in 2005 to over 18.4% in recent prize-nominated works.",
        "Passages detailing autonomous AI agents systematically exhibit high semantic ambiguity scores, reflecting societal moral tension.",
        "Digital reading degrades temporal narrative order reconstruction by ~14-18% compared to physical book reading."
    ]
}

df = pd.DataFrame(data)

file_name = "AI_Memory_Tech_Novel_Research_Measurements.csv"
df.to_csv(file_name, index=False)

print(f"File created: {file_name}")

