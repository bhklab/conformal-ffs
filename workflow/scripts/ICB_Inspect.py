import pyreadr
import os
import pandas as pd

ICB_dir = "data/procdata/ICB/"

# View one ICB data
ICB_Liu = "data/procdata/ICB/ICB_Liu_filtered.RData"
output = pyreadr.read_r(ICB_Liu)
    # each file contains two dataframes: clinical and expr with matched patient IDs
clinical_df = output.get('clinical', None)
rna_df = output.get('expr', None)


# generate summary for all ICB data by recist and response
summary_data = []

# Iterate over files in the directory
for file_name in os.listdir(ICB_dir):
    if file_name.endswith(".RData"):
        file_path = os.path.join(ICB_dir, file_name)
        
        try:
            result = pyreadr.read_r(file_path)

            clinical_df = result.get('clinical', None)
            rna_df = result.get('expr', None)

            if clinical_df is None or rna_df is None:
                print(f"Missing keys in {file_name}")
                continue

            # Get counts of 'recist' and 'response'
            recist_counts = clinical_df['recist'].value_counts().to_dict()
            data_no_recist = clinical_df[clinical_df['recist'].isna()]

            response_counts = clinical_df['response'].value_counts().to_dict()

            cancer_type = ', '.join(clinical_df['cancer_type'].dropna().unique())
            treatment = ', '.join(clinical_df['treatment'].dropna().unique())
            rna = ', '.join(clinical_df['rna'].dropna().unique())
            total_patients = clinical_df.shape[0]

            # Store summary
            summary_data.append({
                'File': file_name,
                'Cancer_Type': cancer_type,
                'Treatment': treatment,
                'RNA': rna,
                'Total_Patients': total_patients,
                'Recist_Counts': recist_counts,
                'Response_Counts': response_counts,
            })

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.join(summary_df.pop('Recist_Counts').apply(pd.Series).add_prefix('Recist_'))
summary_df = summary_df.join(summary_df.pop('Response_Counts').apply(pd.Series).add_prefix('Response_'))

summary_df