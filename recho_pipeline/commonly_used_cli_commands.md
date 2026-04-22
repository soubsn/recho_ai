# Full ESC-50, both x(t) and y(t), exported alongside the .npy cache.
python data/sample_data.py \
    --source esc50 \
    --max-clips-per-class -1 \
    --xy \
    --export-dir /Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text/

# 1. Mixtures only
python -m recho_pipeline.data.mix_esc50 --no-originals

# 2. Hopf-integrate just the mixtures
python recho_pipeline/data/sample_data.py --source esc50 \
    --csv-name esc50_mixed.csv --audio-subdir audio_mixed \
    --max-clips-per-class -1 \
    --export-dir /Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text_mix_only

# 3. Merge with the existing originals cache
python -m recho_pipeline.data.merge_hopf_text \
    --source /Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text \
    --source /Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text_mix_only \
    --out-dir /Users/nic-spect/data/recho_ai/Kaggle_Environmental_Sound_Classification_50/hopf_text_combined


