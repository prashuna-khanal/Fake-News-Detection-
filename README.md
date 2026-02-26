# AI-Based Fake News Detection for Nepal's Digital News Ecosystem

A Random Forest classifier to detect misinformation in Nepali digital news, combining Nepal-specific data with Kaggle-style datasets.


## Run the Pipeline

```bash
# Step 1: Generate dataset
python src/generate_dataset.py

# Step 2: Preprocess data
python src/preprocess.py

# Step 3: Train model
python src/train_model.py

# Step 4: Evaluate model
python src/evaluate_model.py

# Step 5: Predict (CLI)
python src/predict.py
```

## Project Structure

```
Fake News Detection/
├── data/
│   ├── raw/                    # Raw CSV datasets
│   └── processed/              # Combined & cleaned dataset
├── models/                     # Saved model & vectorizer
├── outputs/                    # Evaluation plots & reports
├── src/                        # Source code
└── requirements.txt
```

## Dataset

- **Nepal Real News**: ~300 samples from The Himalayan Times, BBC Nepali, Gorkhapatra
- **Nepal Fake News**: ~300 samples from social media, viral posts, fact-checking sites
- **Kaggle-style Data**: ~1000 samples (WELFake/ISOT inspired)
- **Total**: ~1600 labeled samples (0 = Fake, 1 = Real)

## Model

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF (unigrams + bigrams, top 5000 features)
- **Tuning**: GridSearchCV with 5-fold cross-validation
- **Expected Accuracy**: ~87–92%
