import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot directory
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Data path
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_credit_card_churn_data.csv')

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        print("Cleaned data loaded for EDA.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Cleaned data file not found at {DATA_PATH}")

def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'age_distribution.png'))
    plt.close()

def plot_tenure_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Tenure', color='purple')
    plt.title('Tenure Distribution')
    plt.xlabel('Tenure (Years)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'tenure_distribution.png'))
    plt.close()

def plot_churn_count(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', color='salmon')
    plt.title('Churn vs Non-Churn Count')
    plt.xticks([0, 1], ['No Churn', 'Churn'])
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'churn_count.png'))
    plt.close()

def plot_card_type_vs_churn(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='HasCrCard', hue='Churn', palette='Set1')
    plt.title('Credit Card Possession vs Churn')
    plt.xticks([0, 1], ['No Card', 'Has Card'])
    plt.xlabel('Has Credit Card')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'card_vs_churn.png'))
    plt.close()

def plot_isactive_vs_churn(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='IsActiveMember', hue='Churn', palette='coolwarm')
    plt.title('Active Membership vs Churn')
    plt.xticks([0, 1], ['Inactive', 'Active'])
    plt.xlabel('Is Active Member')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'active_vs_churn.png'))
    plt.close()

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    # Drop non-numeric columns (CustomerID, Gender)
    df_numeric = df.select_dtypes(include=['number'])
    corr = df_numeric.corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'correlation_matrix.png'))
    plt.close()


# def plot_balance_vs_churn(df):
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=df, x='Churn', y='Balance', palette='Pastel1')
#     plt.title('Balance Distribution vs Churn')
#     plt.xticks([0, 1], ['No Churn', 'Churn'])
#     plt.ylabel('Balance')
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, 'balance_vs_churn.png'))
#     plt.close()
def plot_balance_vs_churn():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define data and plot paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BALANCE_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'cleaned_credit_card_churn_data.csv')
    BALANCE_PLOT_PATH = os.path.join(BASE_DIR, '..', 'reports', 'plots', 'balance_vs_churn.png')

    # Load cleaned data
    balance_df = pd.read_csv(BALANCE_DATA_PATH)

    # Plot balance vs churn using raw balance
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=balance_df, x='Churn', y='Balance', palette='Pastel1')
    plt.title('Balance Distribution vs Churn')
    plt.xticks([0, 1], ['No Churn', 'Churn'])
    plt.ylabel('Balance')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(BALANCE_PLOT_PATH), exist_ok=True)
    plt.savefig(BALANCE_PLOT_PATH)
    plt.close()

    print(f"Plot saved to {BALANCE_PLOT_PATH}")


def plot_churn_by_age_group(df):
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70, 90], labels=["18–30", "31–40", "41–50", "51–60", "61–70", "71+"])
    age_group_churn = df.groupby('AgeGroup')['Churn'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=age_group_churn, x='AgeGroup', y='Churn', color='teal')
    plt.title("Churn Rate by Age Group")
    plt.ylabel("Churn Rate")
    plt.xlabel("Age Group")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'churn_by_age_group.png'))
    plt.close()



def run_eda():
    df = load_data()
    plot_age_distribution(df)
    plot_tenure_distribution(df)
    plot_churn_count(df)
    plot_card_type_vs_churn(df)
    plot_isactive_vs_churn(df)
    plot_correlation_matrix(df)
    plot_balance_vs_churn()
    plot_churn_by_age_group(df)
    print("EDA completed. All plots saved to reports/plots/.")

if __name__ == "__main__":
    run_eda()
