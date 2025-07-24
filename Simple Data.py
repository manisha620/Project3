{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b6fbddb1-7e4e-4493-95b0-57c516d6f037",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-24 13:49:03.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.009 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.010 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.010 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.037 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.037 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.039 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.040 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.052 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.083 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      " [[12  1]\n",
      " [ 5 22]]\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "         Low       0.71      0.92      0.80        13\n",
      "      Medium       0.96      0.81      0.88        27\n",
      "\n",
      "    accuracy                           0.85        40\n",
      "   macro avg       0.83      0.87      0.84        40\n",
      "weighted avg       0.88      0.85      0.85        40\n",
      "\n",
      "\n",
      "Top Factors Driving CLV:\n",
      "\n",
      "              Feature  Importance\n",
      "2     avg_order_value    0.415029\n",
      "1  purchase_frequency    0.380042\n",
      "3  last_purchase_days    0.078920\n",
      "0          CustomerID    0.068387\n",
      "4              tenure    0.057621\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-24 13:49:03.269 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:03.270 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Importing libraries\n",
    "import pandas as pd\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Sample dataset\n",
    "df = pd.read_csv(\"sample_customer_data.csv\")  # Must contain 'features' & 'target' columns\n",
    "\n",
    "# data cleaning\n",
    "df['CustomerID'] = df['CustomerID'].str.replace(\"C\",'')\n",
    "\n",
    "\n",
    "# Splitting data\n",
    "X = df.drop(\"target\", axis=1)\n",
    "y = df[\"target\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
    "\n",
    "# Training model\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train, y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "feature_importance = pd.DataFrame({\n",
    "    \"Feature\": X.columns,\n",
    "    \"Importance\": model.feature_importances_\n",
    "}).sort_values(\"Importance\", ascending=False)\n",
    "\n",
    "print(\"\\nTop Factors Driving CLV:\\n\")\n",
    "print(feature_importance.head(5))\n",
    "\n",
    "# Recommendation Logic\n",
    "def business_insight(feature, value):\n",
    "    if feature == \"purchase_frequency\" and value > 10:\n",
    "        return \"Encourage loyalty upgrades\"\n",
    "    elif feature == \"avg_order_value\" and value < 100:\n",
    "        return \"Bundle products to raise order value\"\n",
    "    else:\n",
    "        return \"Maintain engagement strategy\"\n",
    "    \n",
    "\n",
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "st.title(\"CLV Evaluation Dashboard\")\n",
    "\n",
    "# Interactive Visualization\n",
    "st.subheader(\"Feature Importance\")\n",
    "st.dataframe(feature_importance)\n",
    "\n",
    "st.subheader(\"Model Performance\")\n",
    "st.text(\"Classification Report\")\n",
    "st.text(classification_report(y_test, y_pred))\n",
    "\n",
    "st.subheader(\"Confusion Matrix\")\n",
    "fig, ax = plt.subplots()\n",
    "ax.matshow(confusion_matrix(y_test, y_pred), cmap=\"Blues\")\n",
    "st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ed894e8e-3777-4699-bd95-45f48b013ffb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Top Factors Driving CLV:\n",
      "\n",
      "              Feature  Importance\n",
      "2     avg_order_value    0.415029\n",
      "1  purchase_frequency    0.380042\n",
      "3  last_purchase_days    0.078920\n",
      "0          CustomerID    0.068387\n",
      "4              tenure    0.057621\n"
     ]
    }
   ],
   "source": [
    "feature_importance = pd.DataFrame({\n",
    "    \"Feature\": X.columns,\n",
    "    \"Importance\": model.feature_importances_\n",
    "}).sort_values(\"Importance\", ascending=False)\n",
    "\n",
    "print(\"\\nTop Factors Driving CLV:\\n\")\n",
    "print(feature_importance.head(5))\n",
    "\n",
    "# Recommendation Logic\n",
    "def business_insight(feature, value):\n",
    "    if feature == \"purchase_frequency\" and value > 10:\n",
    "        return \"Encourage loyalty upgrades\"\n",
    "    elif feature == \"avg_order_value\" and value < 100:\n",
    "        return \"Bundle products to raise order value\"\n",
    "    else:\n",
    "        return \"Maintain engagement strategy\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b6eb1009-43a4-42cc-b8c9-0e47ce2f4fd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-24 13:49:16.036 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.037 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.040 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.043 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.043 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.044 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.045 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.046 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.047 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.056 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.058 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.059 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.074 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.274 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-24 13:49:16.275 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "st.title(\"CLV Evaluation Dashboard\")\n",
    "\n",
    "# Interactive Visualization\n",
    "st.subheader(\"Feature Importance\")\n",
    "st.dataframe(feature_importance)\n",
    "\n",
    "st.subheader(\"Model Performance\")\n",
    "st.text(\"Classification Report\")\n",
    "st.text(classification_report(y_test, y_pred))\n",
    "\n",
    "st.subheader(\"Confusion Matrix\")\n",
    "fig, ax = plt.subplots()\n",
    "ax.matshow(confusion_matrix(y_test, y_pred), cmap=\"Blues\")\n",
    "st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a85e1c62-6236-4421-b712-da891296d116",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
