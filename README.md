**README.md** begin with 3 bullet points. Each bullet is no more than 50 words.

- **An explanation of how you scraped the data:** <br><br>
The code uses GitHub's API to scrape user profiles from Sydney with over 100 followers. It fetches user details (e.g., login, company, bio) and repositories, handling rate limits with time.sleep(). Data is saved into users.csv and repositories.csv after organizing and cleaning fields like company names and licenses.
   
- **The most interesting and surprising fact you found after analyzing the the data:** <br><br>
My analysis unveils a fascinating trend that Sydney developers’ repositories with the MIT license and languages like Go and JavaScript attract the highest engagement. This insight reveals how flexible licensing and strategic language choices can boost visibility and popularity, making these factors powerful tools for attracting more interaction.
   
- **An actionable recommendation for developers based on your analysis:**<br><br>
To maximize visibility and engagement, developers should license projects under MIT for its openness and consider coding in high-demand languages like Go and JavaScript. This combination not only attracts more contributors but also increases project appeal in the broader developer community.
   

## Analyzing Engagement Factors for GitHub Repositories in Sydney ##

### Key Objective ###

The primary objective of this analysis was to identify the most influential features—such as programming language, license type, and creation date—that contribute to repositories surpassing an engagement threshold of 100 stargazers. This threshold represents a repository's level of popularity in the GitHub community.

### Project Setup and Analysis Steps ###

- **Important import libraries:**
  - `train_test_split`: Splits data into training and testing sets.
  - `LogisticRegression`: Builds a logistic regression model to predict outcomes.
  - `classification_report`: Generates a performance summary of the model.
  - `numpy`: Used for numerical operations (although not directly here).

- **Setting Engagement** Threshold: Define an engagement threshold of `100` stargazers, meaning any repository with over 100 stargazers will be considered `"high engagement"`.

- **Creating a Binary Target Variable:** Add a new column `high_engagement` to `repositories_df`. Set it to 1 if `stargazers_count` is above 100, otherwise 0. This will be our target for predicting high engagement.

- **Selecting and Preparing Features:** Create a subset `repos_features` with relevant columns (`language`, `license_name`, `created_at`) and convert created_at into a new `year_created` column to capture when each repository was created.

- **Encoding Categorical Variables:** Convert categorical features (language, license_name, year_created) into dummy/indicator variables (0s and 1s). `drop_first=True` reduces multicollinearity by dropping one category per feature.

- **Defining Target Variable:** Set `target` to the binary high_engagement column, representing the outcome the model will predict.

- **Splitting Data into Training and Testing Sets:** Split data into training (70%) and testing (30%) sets. `random_state=42` ensures reproducibility.

- **Training the Logistic Regression Model:** Initialize and trains a logistic regression model with a maximum of `200` iterations, fitting it on the training data (`X_train`, `y_train`).

- **Extracting Top Features:** Create a DataFrame to store each feature and its impact (`Coefficient`) on predicting high engagement. `nlargest(3, 'Coefficient')` selects the top three most influential features.

- **Evaluating the Model:** Predict y_test using X_test and generates a report (`classification_report_data`) that shows model accuracy, precision, recall, and F1 score.

- **Displaying Top Features:** Uses `ace_tools` to display the top 3 predictive features for high engagement in a user-friendly format.


### Complete Code ###

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

engagement_threshold = 100

repositories_df['high_engagement'] = (repositories_df['stargazers_count'] > engagement_threshold).astype(int)

# Features selected : language, license presence, and repository creation year.
repos_features = repositories_df[['language', 'license_name', 'created_at']].copy()
repos_features['year_created'] = pd.to_datetime(repositories_df['created_at']).dt.year

repos_features = pd.get_dummies(repos_features[['language', 'license_name', 'year_created']], drop_first=True)

target = repositories_df['high_engagement']

X_train, X_test, y_train, y_test = train_test_split(repos_features, target, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Identify the most impactful feature.
coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': log_reg.coef_[0]})
top_features = coef_df.nlargest(3, 'Coefficient')

y_pred = log_reg.predict(X_test)
classification_report_data = classification_report(y_test, y_pred, output_dict=True)

# Top 3 feature most likely to cross the engagement threshold
import ace_tools as tools; tools.display_dataframe_to_user(name="Top 3 Engagement-Predictive Features", dataframe=top_features)

top_features, classification_report_data

Editing TDS-Project-1/README.md at main · 23f3004236/TDS-Project-1
```
### Result Table ###
Below are the top three features most associated with crossing the engagement threshold (100 stargazers):

| Rank | Feature          | Coefficient |
|------|------------------|-------------|
| 1    | MIT License      | 1.004537    |
| 2    | Go Language      | 0.314887    |
| 3    | JavaScript       | 0.218959    |

### Findings ###

The model’s results reveal that license type and language choice are the strongest predictors of high engagement for repositories. The analysis suggests that the ***MIT license*** and specific languages like ***Go*** and ***JavaScript*** enhance repository visibility and popularity, especially among the Sydney GitHub community. This analysis underscores how license flexibility and language preference can impact repository engagement. By making informed choices about licensing and technology stack, developers can potentially attract more visibility and interaction for their projects.


