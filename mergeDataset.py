import pandas as pd

def merge_datasets(files, output_file):
    merged_data = pd.DataFrame()
    for file in files:
        data = pd.read_csv(file, encoding='unicode_escape')

        important_cols = ['text', 'sentiment']
        data = data.dropna(subset=important_cols).copy()

        for col in ['text', 'sentiment']:
            data[col] = data[col].fillna('') if col in data.columns else ''

        data = data[['text', 'sentiment']].dropna(subset=['text', 'sentiment'])
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    merged_data.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")


if __name__ == "__main__":
    files = ['sentiment_analysis.csv', 'test.csv']
    output_file = 'merged_sentiment_analysis.csv'
    merge_datasets(files, output_file)