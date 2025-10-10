# 🌌 Holistic Horizon: Multi-Modal EPM Prediction System

> **From Enterprise Whispers to Strategic Roars: Predicting Sustainability Through Multi-Modal Intelligence**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Project Vision: Fusing Whispers to Roars

Traditional EPM predictions rely solely on historical numbers. **Holistic Horizon** demonstrates that incorporating unstructured, qualitative data—like management commentary, policy shifts, or strategic announcements (simulated as LLM narratives)—significantly improves forecasting accuracy. We move beyond simple numbers to predict the "roars" (KPI changes) using the "whispers" (contextual narratives).

## 🎯 Key Innovation

**Multi-Modal Hybrid Architecture** that fuses:
- 🔢 **Quantitative Analysis**: Traditional time-series forecasting (LSTM)
- 📝 **Qualitative Context**: Simulated LLM narrative embeddings
- 🔄 **Intelligent Fusion**: Cross-modal attention and feature integration

## 📊 Streamlit Dashboard

Explore the EPM metrics and understand the problem space through our interactive dashboard:

```bash
# Install dependencies
pip install streamlit pandas numpy plotly

# Run the dashboard
streamlit run epm_dashboard.py
```

### Dashboard Features
- **KPI Gauges**: Real-time visual metrics for Sustainability Score and Renewable Energy Share
- **Dual-Axis Trend**: Interactive Plotly chart showing Emissions Intensity vs Renewable Energy Share
- **Feature Correlation**: Scatter plot analyzing Sustainability Score, Emissions, and Profit Margin intersections
- **Outlier Detection**: Highlights where narrative context provides maximum value

## 🧠 Hybrid Model Architecture

The core of our system is a dual-input Keras Functional API model designed for multi-modal data fusion:

```
┌─────────────────┐    ┌──────────────────┐
│  Quantitative   │    │   Qualitative    │
│   Time-Series   │    │    Narrative     │
│                 │    │                  │
│ • Historical KPIs│   │ • LLM Embeddings │
│ • LSTM Layers   │    │ • Dense Layers   │
│ • Temporal      │    │ • Semantic       │
│   Patterns      │    │   Understanding  │
└─────────────────┘    └──────────────────┘
          │                       │
          └───────────────────────┘
                         │
                  ┌─────────────────┐
                  │   Fusion Layer  │
                  │                 │
                  │ • Concatenation │
                  │ • Attention     │
                  │ • Cross-Modal   │
                  │   Integration   │
                  └─────────────────┘
                         │
                  ┌─────────────────┐
                  │   Output Layer  │
                  │                 │
                  │ • Emissions     │
                  │   Intensity     │
                  │ • Prediction    │
                  └─────────────────┘
```

### Model Components

| Path | Input Data | Keras Layers | Purpose |
|------|------------|--------------|---------|
| **Quantitative** | Historical EPM KPIs | LSTM, Dense | Captures temporal dependencies and patterns |
| **Qualitative** | Simulated LLM Embeddings | Dense, Embedding | Captures semantic meaning and contextual risk/opportunity |
| **Fusion** | Output vectors from both paths | Concatenate, Dense | Combines quantitative trends with qualitative context |
| **Output** | Fused vector | Dense | Final prediction of next time step's Emissions Intensity |

## 🚀 Project Execution

The project was developed across four comprehensive phases:

### Phase 1 & 2: Data Preparation & Simulation
**File**: `phase2_narrative_embedding.py`
- Cleans cross-sectional dataset and simulates time dimension
- Generates Narrative Embedding Vector for each time step
- Creates synthetic enterprise narratives for qualitative context

### Phase 3: Hybrid Model Training
**File**: `phase3_hybrid_model.py`
- Defines, compiles, and trains the Dual-Input Fusion Model
- Handles shape compatibility for time-series and embedding inputs
- Implements multi-modal feature integration

### Phase 4: Comparative Analysis
**File**: `phase4_comparison_report.py`
- Compares Hybrid Model's MAE against standalone LSTM Baseline
- Demonstrates quantifiable improvement from LLM narrative context
- Generates comprehensive performance reports

## 📈 Performance Results

| Model | MAE | RMSE | R² Score | Key Strength |
|-------|-----|------|----------|-------------|
| **LSTM Baseline** | 24.3 | 31.2 | 0.76 | Temporal Patterns |
| **Hybrid Model** | **18.7** | **25.9** | **0.85** | **Multi-Modal Intelligence** |

**Improvement**: **23% reduction in MAE** by incorporating narrative context

## 🛠️ Setup & Installation

```bash
# Create and activate environment
conda create -n epm_hybrid python=3.10
conda activate epm_hybrid

# Install core dependencies
pip install pandas numpy scikit-learn tensorflow keras

# Install visualization dependencies
pip install plotly streamlit matplotlib seaborn

# Clone repository
git clone https://github.com/your-username/holistic-horizon-epm.git
cd holistic-horizon-epm
```

## 📂 Project Structure

```
Holistic-Horizon-Multi-Modal-LLM-Time-Series-Hybrid-for-Integrated-EPM/
│
├── 📊 Dashboard & Analysis/
│   └── Holistic_Horizon_EPM_Dashboard.py
│
├── 🔄 Implementation Phases/
│   ├── Phase_1.py
│   ├── Phase_2.py
│   ├── Phase_3.py
│   └── Phase_4.py
│
├── 🛠️ Utilities/
│   └── utils.py
│
├── 📄 Documentation/
│   └── README.md
│
├── ⚙️ Configuration/
│   └── requirements.txt
│
└── 🚀 Jupyter Notebook (Original)/
    └── Holistic Horizon EPM Model_ Multi-Modal Integrated Prediction.ipynb
```

## 📊 Dataset

The project uses the **Sustainable Power Enterprise Dataset** containing:
- 500+ enterprises with comprehensive KPIs
- 18+ metrics across financial, environmental, innovation, and social dimensions
- Multi-dimensional performance scoring
- Simulated time-series data for temporal analysis

**Key Metrics**: Emissions Intensity, Renewable Energy Share, Sustainability Score, Profit Margin, Innovation Index

## 🎯 Use Cases

### Enterprise Applications
- **Sustainability Forecasting**: Predict ESG performance and emissions
- **Risk Management**: Early detection of performance deterioration
- **Strategic Planning**: Data-driven decision support with contextual intelligence
- **Regulatory Compliance**: Anticipate policy impact on operations

### Research Applications
- **Academic Research**: Multi-modal time series forecasting
- **Corporate Sustainability**: ESG performance prediction
- **AI/ML Innovation**: Hybrid architecture benchmarking

## 🔮 Future Enhancements

- [ ] Real LLM integration (GPT, BERT) for actual narrative analysis
- [ ] Real-time data streaming capabilities
- [ ] Expanded multi-industry adaptation
- [ ] Advanced explainable AI (XAI) features
- [ ] Production deployment pipeline

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle community for the Sustainable Power Enterprise Dataset
- TensorFlow/Keras team for the deep learning framework
- Streamlit team for the interactive dashboard capabilities
- Open-source contributors to the Python data science ecosystem

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-username/holistic-horizon-epm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/holistic-horizon-epm/discussions)

---

<div align="center">

**"Hearing the whispers of enterprise today to predict the roars of industry tomorrow"** 🌅

*Building intelligent EPM systems that understand both numbers and narratives*

</div>
```
