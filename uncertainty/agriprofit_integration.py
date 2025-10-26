"""
AGRIPROFIT INTEGRATION GUIDE
How to add Ultimate Time Series Analysis to your existing system

=============================================================================
STEP 1: File Structure
=============================================================================

Your project structure should look like this:

AgriProfit/
├── accounting/
├── analysis/
├── data/
├── decision/
├── features/
├── forecast/
├── models/
├── ui/
│   └── app_streamlit.py
├── uncertainty/
├── ultimate_time_series_analysis.py  ← ADD THIS (the main analyzer)
└── ultimate_visualizer.py             ← ADD THIS (the visualizer)

=============================================================================
STEP 2: Update app_streamlit.py
=============================================================================
"""

# Add these imports at the top of ui/app_streamlit.py
from ultimate_visualizer import create_streamlit_tab

# ============================================================================
# OPTION A: Add as a new tab (RECOMMENDED)
# ============================================================================

# Find this line in your app_streamlit.py:
# tabs = st.tabs(["Data Analysis", "Prices & Forecast", "Profit Scenarios", "Accounting", "Risk & Advice", "Quality"])

# Replace it with:
tabs = st.tabs([
    "Data Analysis", 
    "Prices & Forecast", 
    "Profit Scenarios", 
    "Accounting", 
    "Risk & Advice", 
    "Quality",
    "🔬 Ultimate Analysis"  # ← NEW TAB
])

# Then at the end, add:
with tabs[6]:  # The new Ultimate Analysis tab
    create_streamlit_tab(prices)


# ============================================================================
# OPTION B: Add to existing "Data Analysis" tab
# ============================================================================

# Find your "Data Analysis" tab content and add at the end:

with tabs[0]:  # Data Analysis tab
    st.subheader("Data Analysis")
    
    # ... your existing analysis code ...
    
    # Add separator
    st.markdown("---")
    
    # Add Ultimate Analysis
    st.subheader("🔬 Ultimate Time Series Analysis")
    
    if st.button("Run Ultimate Analysis", type="primary"):
        from ultimate_visualizer import integrate_with_agriprofit
        
        with st.spinner("Running comprehensive analysis..."):
            output = integrate_with_agriprofit(prices)
        
        st.success("✅ Analysis complete!")
        
        # Display dashboard
        st.pyplot(output['figure'])
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            with open('agriprofit_ultimate_analysis.png', 'rb') as f:
                st.download_button(
                    "📥 Download Dashboard",
                    f,
                    file_name="ultimate_analysis.png",
                    mime="image/png"
                )
        with col2:
            with open('agriprofit_ultimate_report.pdf', 'rb') as f:
                st.download_button(
                    "📄 Download PDF Report",
                    f,
                    file_name="ultimate_report.pdf",
                    mime="application/pdf"
                )


# ============================================================================
# OPTION C: Standalone page (Advanced)
# ============================================================================

# Create a new file: ui/ultimate_analysis_page.py

import streamlit as st
from ultimate_visualizer import integrate_with_agriprofit
from data.loaders import load_price_csv

st.set_page_config(page_title="Ultimate Analysis", layout="wide")

st.title("🔬 Ultimate Time Series Analysis")

# File upload
uploaded = st.file_uploader("Upload your price data", type=['csv'])
prices = load_price_csv(uploaded)

if st.button("Run Analysis", type="primary"):
    with st.spinner("Analyzing..."):
        output = integrate_with_agriprofit(prices)
    
    st.pyplot(output['figure'])


# ============================================================================
# STEP 3: Simple Command-Line Usage (No Streamlit)
# ============================================================================

"""
If you just want to run analysis from command line:

python run_ultimate_analysis.py --input prices.csv --output results/
"""

# Create run_ultimate_analysis.py:
import argparse
import pandas as pd
from ultimate_visualizer import integrate_with_agriprofit

def main():
    parser = argparse.ArgumentParser(description='Run Ultimate Time Series Analysis')
    parser.add_argument('--input', required=True, help='Input CSV file (date,price)')
    parser.add_argument('--output', default='results/', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    df['date'] = pd.to_datetime(df['date'])
    
    # Run analysis
    output = integrate_with_agriprofit(df)
    
    print(f"\n✓ Results saved to {args.output}")

if __name__ == "__main__":
    main()


# ============================================================================
# STEP 4: Using with Existing Analysis Results
# ============================================================================

"""
You can also use the analyzer programmatically in your existing code:
"""

def enhanced_data_analysis(prices_df):
    """
    Drop-in replacement for your existing analyze_data functions
    """
    from ultimate_time_series_analysis import UltimateTimeSeriesAnalyzer
    
    # Create analyzer
    analyzer = UltimateTimeSeriesAnalyzer(prices_df, frequency='M')
    
    # Run full analysis
    results = analyzer.run_ultimate_analysis()
    
    # Access specific results
    hurst = results['nonlinear']['hurst_exponent']['rs_hurst']
    best_arima = results['arima_identification']['best_models'][0]
    is_stationary = results['stationarity_advanced']['conclusion']
    
    return {
        'hurst': hurst,
        'arima_model': best_arima['order'],
        'arima_aic': best_arima['aic'],
        'stationarity': is_stationary,
        'full_results': results
    }


# ============================================================================
# STEP 5: Custom Analysis Selection
# ============================================================================

"""
If you only want specific analyses (not all 15 modules):
"""

from ultimate_time_series_analysis import UltimateTimeSeriesAnalyzer

def run_selective_analysis(prices_df, analyses=['distribution', 'trend', 'spectral']):
    """
    Run only selected analyses
    
    Available analyses:
    - distribution_advanced
    - trend_advanced
    - spectral
    - nonlinear
    - stationarity_advanced
    - identify_arima_models
    - detect_structural_breaks_advanced
    - analyze_extreme_values
    - analyze_information_content
    """
    
    analyzer = UltimateTimeSeriesAnalyzer(prices_df)
    results = {}
    
    if 'distribution' in analyses:
        results['distribution'] = analyzer.analyze_distribution_advanced()
    
    if 'trend' in analyses:
        results['trend'] = analyzer.analyze_trend_advanced()
    
    if 'spectral' in analyses:
        results['spectral'] = analyzer.analyze_spectral()
    
    if 'nonlinear' in analyses:
        results['nonlinear'] = analyzer.analyze_nonlinear()
    
    if 'stationarity' in analyses:
        results['stationarity'] = analyzer.analyze_stationarity_advanced()
    
    if 'arima' in analyses:
        results['arima'] = analyzer.identify_arima_models()
    
    if 'breaks' in analyses:
        results['breaks'] = analyzer.detect_structural_breaks_advanced()
    
    if 'extreme' in analyses:
        results['extreme'] = analyzer.analyze_extreme_values()
    
    if 'information' in analyses:
        results['information'] = analyzer.analyze_information_content()
    
    return results


# ============================================================================
# STEP 6: Export Results to Excel/CSV
# ============================================================================

def export_results_to_excel(results, filename='ultimate_analysis_results.xlsx'):
    """
    Export all analysis results to Excel with multiple sheets
    """
    import pandas as pd
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Distribution
        if 'distribution_advanced' in results:
            dist = results['distribution_advanced']
            df_dist = pd.DataFrame([dist['moments']])
            df_dist.to_excel(writer, sheet_name='Distribution', index=False)
        
        # Spectral
        if 'spectral' in results:
            spec = results['spectral']
            if spec['dominant_frequencies']:
                df_spec = pd.DataFrame(spec['dominant_frequencies'])
                df_spec.to_excel(writer, sheet_name='Spectral', index=False)
        
        # ARIMA
        if 'arima_identification' in results:
            arima = results['arima_identification']
            if arima['best_models']:
                df_arima = pd.DataFrame(arima['best_models'])
                df_arima.to_excel(writer, sheet_name='ARIMA', index=False)
        
        # Non-linear
        if 'nonlinear' in results:
            nl = results['nonlinear']
            df_nl = pd.DataFrame([{
                'hurst': nl['hurst_exponent']['rs_hurst'],
                'interpretation': nl['hurst_exponent']['interpretation'],
                'lyapunov': nl.get('lyapunov_exponent'),
                'dfa_alpha': nl.get('dfa_alpha'),
            }])
            df_nl.to_excel(writer, sheet_name='NonLinear', index=False)
    
    print(f"✓ Results exported to {filename}")


# ============================================================================
# STEP 7: Batch Processing Multiple Files
# ============================================================================

def batch_analyze_multiple_datasets(file_list):
    """
    Analyze multiple price datasets and compare
    """
    from ultimate_time_series_analysis import UltimateTimeSeriesAnalyzer
    import pandas as pd
    
    comparison = []
    
    for filename in file_list:
        print(f"Analyzing {filename}...")
        
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        
        analyzer = UltimateTimeSeriesAnalyzer(df)
        results = analyzer.run_ultimate_analysis()
        
        # Extract key metrics
        comparison.append({
            'filename': filename,
            'n_observations': analyzer.n,
            'hurst': results['nonlinear']['hurst_exponent']['rs_hurst'],
            'stationarity': results['stationarity_advanced']['conclusion'],
            'best_arima': str(results['arima_identification']['best_models'][0]['order']),
            'predictability': results['information']['predictability']['interpretation'],
        })
    
    # Create comparison dataframe
    df_comparison = pd.DataFrame(comparison)
    df_comparison.to_csv('batch_comparison.csv', index=False)
    
    print("\n✓ Batch analysis complete!")
    print(df_comparison)
    
    return df_comparison


# ============================================================================
# COMPLETE EXAMPLE: Full Integration
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Example 1: Basic usage
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Generate sample data
    dates = pd.date_range('2019-01-01', periods=60, freq='MS')
    prices = 15 + 0.15 * np.arange(60) + 3 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.randn(60)
    df = pd.DataFrame({'date': dates, 'price': prices})
    
    # Run analysis
    from ultimate_visualizer import integrate_with_agriprofit
    output = integrate_with_agriprofit(df)
    
    print("\n✓ Analysis complete!")
    print(f"Hurst exponent: {output['results']['nonlinear']['hurst_exponent']['rs_hurst']:.3f}")
    
    
    # Example 2: Selective analysis
    print("\n" + "="*80)
    print("EXAMPLE 2: Selective Analysis")
    print("="*80)
    
    results = run_selective_analysis(df, analyses=['spectral', 'nonlinear', 'arima'])
    print("\n✓ Selected analyses complete!")
    
    
    # Example 3: Export to Excel
    print("\n" + "="*80)
    print("EXAMPLE 3: Export to Excel")
    print("="*80)
    
    export_results_to_excel(output['results'])
    
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)