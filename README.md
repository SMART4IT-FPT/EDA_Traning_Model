# Resume Data EDA

## Project Overview

This project performs **exploratory data analysis (EDA)** on a dataset of resumes. The primary goals are:

- Understand the structure and content of the resume text and associated job labels.
- Identify data quality issues such as missing values and duplicates.
- Visualize distributions and patterns within labels and resume text.
---
## Dataset Description

- **File**: `resumes_data.csv`
- **Total records**: 29,783 resumes
- **Columns**:
  - `file_id`: Unique identifier for each resume.
  - `text`: The full resume text (string).
  - `label`: Job-related labels (e.g., `Python_Developer;Web_Developer`), total of **552 unique values**.
- **Grouped Occupations**: Labels are mapped into **10 main job categories** (e.g., `Software_Developer`, `Database_Administrator`).
- **Missing labels**: ~2.51% of resumes have no label.

---
## Exploratory Data Analysis (EDA)

### Data Checks
- Loaded with `pandas`, confirmed **29,783 rows**.
- Found **552 distinct label strings**.
[![top50.png](https://i.postimg.cc/8PHg5RTD/top50.png)](https://postimg.cc/D88MpbWY)

- **Missing labels**: 748 records (~2.51%) had empty label values.
- **Duplicate resumes**: 740 records.
### Label Analysis
Based on the label distribution, it is evident that the "Software Developer" label is the most frequent compared to the others. This indicates a significant class imbalance in the dataset, which could negatively impact the performance of the multi-label classification model.
[![label.png](https://i.postimg.cc/SsRNkgBn/label.png)](https://postimg.cc/tZ0HzNCb)
### Token Position Analysis
Based on the token distribution, it can be observed that this dataset consists of long-form text data. The high token counts in CVs are mainly due to their multi-section structure (e.g., education, experience, skills, projects, etc.).
[![token.png](https://i.postimg.cc/L51D5Yhj/token.png)](https://postimg.cc/0K9DHNYN)
## Conclusion

The exploratory data analysis (EDA) of the resume dataset has provided valuable insights into its structure, quality, and label distribution:
- A **significant class imbalance** was identified, especially the **overrepresentation of the "Software Developer"** category. This issue should be addressed during model training 
- **Token distribution analysis** shows that resumes are typically **long-form documents**, often with high token counts. This has implications for model selection, favoring architectures that support **long context windows**, such as **Longformer** or **BigBird**.

These insights serve as a strong foundation for building effective resume classification or matching models and guide the necessary **preprocessing and modeling strategies** for future development.
