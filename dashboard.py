import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# ==================================
# SETTING PAGE CONFIGURATION
# ==================================

# Set page configuration
st.set_page_config(
    page_title="Cost Breakdown Dashboard",
    layout="wide",
)

# Apply custom CSS for dark theme styling and full-page background image
st.markdown("""
    <style>
    body {
        background-color: #222222;
        color: #FFFFFF;
    }
    .main-content {
        background-color: rgba(0, 0, 0, 0.8);
        padding: 10px;
        border-radius: 10px;
    }
    .dataframe th {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }
    .dataframe td {
        background-color: #222222 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Display title
st.title("Cost Breakdown Dashboard")

# Horizontal tabs at the top of the page
tabs = st.tabs(["Data", "Project Data", "Cost Breakdown Structure",
                "Cost Analysis", "NRM-1_ECB", "Cost Efficiency Evaluation"])

# Content for each tab
with tabs[0]:
    st.header("Data Section")

    # File path and sheet name
    file_path = "C:/Users/danin/OneDrive/2024_2025/3. Projects/cost_breakdown_analysis/Dummy_Database.xlsx"  # Replace with your Excel file path
    sheet_name = "Data"  # Specify the sheet name

    try:
        # Update Pandas option to allow more styled cells
        pd.set_option("styler.render.max_elements", 710486)

        # Read Excel file into DataFrame, skip unwanted rows
        data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)

        # Clean column names (removes unwanted spaces)
        data.columns = data.columns.str.strip()

        # Display DataFrame without using Styler (if cells exceed limit)
        if data.empty:
            st.write("No data available for the selected filters.")
        else:
            st.dataframe(data, use_container_width=True)
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}. Please ensure the path is correct.")
    except Exception as e:
        st.error(f"An error occurred while reading the Excel file: {e}")

with tabs[1]:
    st.header("Project Data Section")

    try:
        # Clean column names
        data.columns = data.columns.str.strip().str.lower()

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Key variables to select
        key_variables = ["contract", "year", "contrat amount", "bua (m2)", "assit class"]

        # Select only the key variables
        selected_data = df[key_variables]

        # Rename columns directly
        selected_data.columns = ['PROJECT', 'YEAR', 'CONTRACT AMOUNT', 'GIFA (m2)', 'CLASSIFICATION']

        # Remove duplicate projects while keeping only the first occurrence
        selected_data = selected_data.drop_duplicates(subset=['PROJECT'])

        # Format contract amount as currency
        selected_data['CONTRACT AMOUNT'] = selected_data['CONTRACT AMOUNT'].apply(lambda x: f"${x:,.2f}")

        # Sort projects based on Project names in ascending order
        selected_data = selected_data.sort_values(by='PROJECT', ascending=True)

        # Display the interactive table
        st.dataframe(selected_data, use_container_width=True)

        st.write("This interactive table allows you to view your selected project data with formatted Contract Amounts and sorted entries.")

    except KeyError as e:
        st.error(f"KeyError: Missing expected column(s): {e}")
    except Exception as e:
        st.error(f"An error occurred while processing and displaying the data: {e}")

# Additional Tabs Logic
with tabs[2]:
    st.header("Cost Breakdown Structure Section")

    try:
        # Create a DataFrame from your data
        cd = pd.DataFrame(data)

        # Key variables to select
        main_variables = ["contract", "bcis (nrm) level 1", "bcis (nrm) level 2", "cost %"]

        # Check if all required columns are present in the data
        missing_columns = [col for col in main_variables if col not in cd.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
        else:
            # Select only the key variables
            cost_data = cd[main_variables]

            # Rename columns for better readability
            cost_data.columns = ['PROJECT', 'BCIS (NRM) LEVEL 1', 'BCIS (NRM) LEVEL 2', 'COST %']

            # Convert 'COST %' column to numeric for calculations
            cost_data['COST %'] = pd.to_numeric(cost_data['COST %'], errors='coerce').fillna(0)

            # Group by Project, Level 1, and Level 2 to sum duplicates
            cost_data = cost_data.groupby(
                ['PROJECT', 'BCIS (NRM) LEVEL 1', 'BCIS (NRM) LEVEL 2'], as_index=False
            ).agg({'COST %': 'sum'})

            # Group by 'PROJECT' and 'BCIS (NRM) LEVEL 1' to calculate total Level 1 Cost %
            level_1_sums = cost_data.groupby(['PROJECT', 'BCIS (NRM) LEVEL 1'], as_index=False)['COST %'].sum()
            level_1_sums.rename(columns={'COST %': 'LEVEL 1 COST %'}, inplace=True)

            # Merge Level 1 totals back into the main dataset
            merged = pd.merge(cost_data, level_1_sums, on=['PROJECT', 'BCIS (NRM) LEVEL 1'], how='left')

            # Create a select box for project filtering
            selected_project = st.selectbox(
                "Select a Project:",
                options=merged['PROJECT'].unique(),
                index=0
            )

            # Filter the data based on the selected project
            project_data = merged[merged['PROJECT'] == selected_project]

            # Prepare the final table structure
            final_table = pd.DataFrame(columns=["Level 1", "Level 2", "Cost %"])
            for level_1 in project_data["BCIS (NRM) LEVEL 1"].unique():
                level_1_data = project_data[project_data["BCIS (NRM) LEVEL 1"] == level_1]
                # Add the Level 1 row
                final_table = pd.concat(
                    [
                        final_table,
                        pd.DataFrame({
                            "Level 1": [f"**{level_1}**"],  # Make Level 1 bold using Markdown
                            "Level 2": [""],
                            "Cost %": [f"{level_1_data['LEVEL 1 COST %'].iloc[0]:.2f}%"]
                        })
                    ],
                    ignore_index=True
                )
                # Add corresponding Level 2 rows
                for _, row in level_1_data.iterrows():
                    if row["BCIS (NRM) LEVEL 2"]:
                        final_table = pd.concat(
                            [
                                final_table,
                                pd.DataFrame({
                                    "Level 1": [""],
                                    "Level 2": [row["BCIS (NRM) LEVEL 2"]],
                                    "Cost %": [f"{row['COST %']:.2f}%"]
                                })
                            ],
                            ignore_index=True
                        )

            # Display the final interactive table
            st.subheader(f"Cost Breakdown Structure Table for {selected_project}")
            st.markdown("""
                <style>
                .dataframe th {
                    background-color: #006699 !important;
                    color: white !important;
                    font-weight: bold;
                }
                .dataframe td {
                    background-color: #f2f2f2 !important;
                    color: black !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # Apply Streamlit Markdown for bold formatting
            st.markdown(final_table.to_markdown(index=False), unsafe_allow_html=True)

            # Additional Notes
            st.write("""
            The table dynamically displays cost data for the selected project, grouping Level 2 values under their corresponding Level 1 values, 
            summing duplicates, and summing percentages at Level 1. This format is interactive and neatly formatted.
            """)

    except KeyError as e:
        st.error(f"KeyError: Missing column(s): {e}. Ensure the dataset contains the required fields.")
    except Exception as e:
        st.error(f"An error occurred while processing and displaying the Cost Breakdown Structure: {e}")
        
# Dashboard code for tab[3]
with tabs[3]:
    st.header("Cost Analysis Section")
    
    try:
        # Create a DataFrame from the provided data
        ca = pd.DataFrame(data)

        # Add a contract selection input
        contracts = ca["contract"].unique()
        selected_contract = st.selectbox("Select a Contract", options=contracts)

        # Filter data based on the selected contract
        filtered_data = ca[ca["contract"] == selected_contract]

        # Group by "bcis (nrm) level 1" and calculate the sum of "cost %"
        grouped_data = filtered_data.groupby("bcis (nrm) level 1", as_index=False)["cost %"].sum()
        grouped_data.columns = ["Level 1", "Total Cost %"]

        # Display the table on the left
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Cost Analysis Table")
            st.dataframe(grouped_data, use_container_width=True)

        # Create a pie chart for "cost %" on the right
        with col2:
            st.write("### Cost Analysis Pie Chart")
            fig = px.pie(grouped_data, names="Level 1", values="Total Cost %", 
                         title=f"Cost Distribution for {selected_contract}")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Ensure the chart only runs inside Tab 4
with tabs[4]:  # NRM-1_ECB Section Tab
    # Streamlit Header for Tab 4
    st.header("NRM-1_ECB Section")

    try:
        @st.cache_data
        def load_data(data):
            nrm = pd.DataFrame(data)

            # Select key variables
            the_variables = ["contract", "bcis (nrm) level 1", "bcis (nrm) level 2", "assit class", "cost %"]
            chosen_data = nrm[the_variables]

            # Modify column names for readability
            chosen_data.columns = ["PROJECT", "BCIS LEVEL 1", "BCIS LEVEL 2", "CLASSIFICATION", "COST %"]
            
            return chosen_data

        # Load data
        chosen_data = load_data(data)

        # Add a multi-select classification filter
        selected_classes = st.multiselect("Select Classification(s)", options=chosen_data["CLASSIFICATION"].unique(), default=chosen_data["CLASSIFICATION"].unique())

        # Filter data based on selected classification(s)
        filtered_nrm_data = chosen_data[chosen_data["CLASSIFICATION"].isin(selected_classes)]

        # Group data by PROJECT and BCIS LEVEL 1
        grouped_df = filtered_nrm_data.groupby(["PROJECT", "BCIS LEVEL 1"])["COST %"].sum().unstack()

        # Define dynamic colors for BCIS LEVEL 1 categories
        unique_levels = grouped_df.columns.unique()
        color_map = plt.cm.get_cmap("tab10", len(unique_levels))
        colors = [color_map(i) for i in range(len(unique_levels))]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        grouped_df.plot(kind='barh', stacked=True, ax=ax, color=colors)

        # Set labels and title
        ax.set_xlabel("Cost %")
        ax.set_ylabel("Project")
        ax.set_title(f"Element Cost Distribution (CLASSIFICATIONS: {', '.join(selected_classes)})")

        # Adjust layout and show legend
        ax.legend(title="BCIS LEVEL 1", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Display the chart in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Ensure the chart only runs inside Tab 4
with tabs[5]:  # Cost Efficiency Section Tab
    # Streamlit Header for Tab 4
    st.header("Cost Efficiency Evaluation")

    try:
        @st.cache_data
        def load_data(data):
            insights = pd.DataFrame(data)

            # Select key variables
            required_variables = ["contract", "cost %", "$/m2"]
            need_data = insights[required_variables]

            # Rename columns for readability
            need_data.columns = ["PROJECT", "COST %", "UNIT COST ($/m²)"]

            return need_data

        # Load data
        need_data = load_data(data)

        # Plot cost efficiency scatter plot
        fig_, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=need_data, x="UNIT COST ($/m²)", y="COST %", hue="PROJECT", palette="tab10", s=100)

        # Set labels and title
        ax.set_xlabel("Unit Cost ($/m²)")
        ax.set_ylabel("Cost %")
        ax.set_title("Cost Efficiency Evaluation Across Projects")

        # Adjust legend position
        ax.legend(title="Project", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Display the plot in Streamlit
        st.pyplot(fig_)

    except Exception as e:
        st.error(f"An error occurred: {e}")