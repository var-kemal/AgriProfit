"""
ULTIMATE TIME SERIES ANALYZER WITH COMPREHENSIVE VISUALIZATIONS
Complete integration with AgriProfit system

Features:
- 15+ analysis modules
- Professional visualizations (20+ plots)
- PDF report generation
- Integration with existing AgriProfit code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Import the analyzer
from ultimate_time_series_analysis import UltimateTimeSeriesAnalyzer


class UltimateVisualizer:
    """
    Comprehensive visualization system for time series analysis
    """
    
    def __init__(self, analyzer):
        """
        Initialize visualizer with analyzer results
        
        Args:
            analyzer: UltimateTimeSeriesAnalyzer instance with completed analysis
        """
        self.analyzer = analyzer
        self.results = analyzer.results
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        
    def create_master_dashboard(self, figsize=(20, 28)):
        """Create comprehensive dashboard with all analyses"""
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(10, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 0: Title and summary
        ax_title = fig.add_subplot(gs[0, :])
        self._plot_title_summary(ax_title)
        
        # Row 1: Price series with trend and regime highlights
        ax_price = fig.add_subplot(gs[1, :])
        self._plot_price_with_trends(ax_price)
        
        # Row 2: Distribution analysis (3 plots)
        ax_dist1 = fig.add_subplot(gs[2, 0])
        ax_dist2 = fig.add_subplot(gs[2, 1])
        ax_dist3 = fig.add_subplot(gs[2, 2])
        self._plot_distribution_suite(ax_dist1, ax_dist2, ax_dist3)
        
        # Row 3: Decomposition (trend, seasonal, residual)
        ax_decomp = fig.add_subplot(gs[3, :])
        self._plot_decomposition(ax_decomp)
        
        # Row 4: Spectral analysis
        ax_spec1 = fig.add_subplot(gs[4, :2])
        ax_spec2 = fig.add_subplot(gs[4, 2])
        self._plot_spectral_analysis(ax_spec1, ax_spec2)
        
        # Row 5: ACF and PACF
        ax_acf = fig.add_subplot(gs[5, 0])
        ax_pacf = fig.add_subplot(gs[5, 1])
        ax_ljung = fig.add_subplot(gs[5, 2])
        self._plot_correlation_suite(ax_acf, ax_pacf, ax_ljung)
        
        # Row 6: Volatility analysis
        ax_vol1 = fig.add_subplot(gs[6, :2])
        ax_vol2 = fig.add_subplot(gs[6, 2])
        self._plot_volatility_analysis(ax_vol1, ax_vol2)
        
        # Row 7: Risk metrics
        ax_risk1 = fig.add_subplot(gs[7, :2])
        ax_risk2 = fig.add_subplot(gs[7, 2])
        self._plot_risk_analysis(ax_risk1, ax_risk2)
        
        # Row 8: Non-linear dynamics
        ax_nl1 = fig.add_subplot(gs[8, 0])
        ax_nl2 = fig.add_subplot(gs[8, 1])
        ax_nl3 = fig.add_subplot(gs[8, 2])
        self._plot_nonlinear_analysis(ax_nl1, ax_nl2, ax_nl3)
        
        # Row 9: Information theory and breaks
        ax_info = fig.add_subplot(gs[9, :2])
        ax_breaks = fig.add_subplot(gs[9, 2])
        self._plot_information_and_breaks(ax_info, ax_breaks)
        
        plt.suptitle('ULTIMATE TIME SERIES ANALYSIS DASHBOARD', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        return fig
    
    def _plot_title_summary(self, ax):
        """Plot title and key summary statistics"""
        ax.axis('off')
        
        # Create summary text
        n = self.analyzer.n
        start = pd.to_datetime(self.analyzer.dates[0]).strftime('%Y-%m-%d')
        end = pd.to_datetime(self.analyzer.dates[-1]).strftime('%Y-%m-%d')
        
        summary_text = f"""
DATASET OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observations: {n}  |  Period: {start} to {end}  |  Frequency: {self.analyzer.frequency}

KEY FINDINGS:
"""
        
        # Add key findings from each module
        findings = []
        
        if 'distribution_advanced' in self.results:
            dist = self.results['distribution_advanced']
            findings.append(f"• Entropy: {dist['information']['shannon_entropy']:.2f} (complexity)")
        
        if 'nonlinear' in self.results:
            nl = self.results['nonlinear']
            hurst = nl['hurst_exponent']['rs_hurst']
            findings.append(f"• Hurst: {hurst:.3f} ({nl['hurst_exponent']['interpretation']})")
        
        if 'stationarity_advanced' in self.results:
            stat = self.results['stationarity_advanced']
            findings.append(f"• Stationarity: {stat['conclusion'].replace('_', ' ').title()}")
        
        if 'arima_identification' in self.results and self.results['arima_identification']['best_models']:
            best = self.results['arima_identification']['best_models'][0]
            findings.append(f"• Best ARIMA: {best['order']} (AIC={best['aic']:.1f})")
        
        if 'spectral' in self.results and self.results['spectral']['dominant_frequencies']:
            dom = self.results['spectral']['dominant_frequencies'][0]
            findings.append(f"• Dominant cycle: {dom['period']:.1f} periods")
        
        summary_text += '\n'.join(findings[:5])  # Show top 5 findings
        
        ax.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))
    
    def _plot_price_with_trends(self, ax):
        """Plot price series with multiple trend lines"""
        dates = self.analyzer.dates
        prices = self.analyzer.prices
        
        # Plot actual prices
        ax.plot(dates, prices, 'o-', linewidth=1.5, markersize=3, 
               label='Actual', color='#2563eb', alpha=0.7)
        
        # Add trends if available
        if 'trend_advanced' in self.results:
            trend = self.results['trend_advanced']
            
            # Linear trend
            if 'linear' in trend['polynomial_models']['degree_1']:
                fitted = trend['polynomial_models']['degree_1']['fitted']
                ax.plot(dates, fitted, '--', linewidth=2, label='Linear Trend', 
                       color='red', alpha=0.7)
            
            # HP filter trend
            if 'hp_filter' in trend:
                hp_trend = trend['hp_filter']['trend']
                ax.plot(dates, hp_trend, '-.', linewidth=2, 
                       label='HP Filter', color='green', alpha=0.7)
            
            # Mark breakpoints
            if 'breakpoints' in trend and trend['breakpoints']:
                for bp in trend['breakpoints'][:3]:  # Show top 3
                    ax.axvline(x=bp['date'], color='orange', 
                             linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_title('Price Series with Trend Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution_suite(self, ax1, ax2, ax3):
        """Plot distribution analysis (histogram, QQ plot, box plot)"""
        prices = self.analyzer.prices
        
        # Histogram with KDE
        ax1.hist(prices, bins=30, alpha=0.6, color='steelblue', 
                edgecolor='black', density=True)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(prices)
        x_range = np.linspace(prices.min(), prices.max(), 100)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        ax1.axvline(np.mean(prices), color='green', linestyle='--', 
                   linewidth=2, label='Mean')
        ax1.axvline(np.median(prices), color='purple', linestyle='--', 
                   linewidth=2, label='Median')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution', fontweight='bold')
        ax1.legend(fontsize=8)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(prices, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Box plot with violin
        parts = ax3.violinplot([prices], positions=[0], widths=0.7,
                               showmeans=True, showmedians=True)
        ax3.set_title('Distribution Shape', fontweight='bold')
        ax3.set_ylabel('Price')
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.3, axis='y')
    
    def _plot_decomposition(self, ax):
        """Plot trend-seasonal decomposition"""
        if 'trend_advanced' in self.results:
            trend = self.results['trend_advanced']
            
            if 'stl_decomposition' in trend and trend['stl_decomposition']:
                stl = trend['stl_decomposition']
                dates = self.analyzer.dates
                
                # Plot observed, trend, seasonal
                ax.plot(dates, self.analyzer.prices, label='Observed', 
                       alpha=0.5, linewidth=1)
                ax.plot(dates, stl.trend, label='Trend', linewidth=2)
                ax.plot(dates, stl.seasonal, label='Seasonal', 
                       linewidth=1.5, alpha=0.7)
                
                ax.set_xlabel('Date', fontweight='bold')
                ax.set_ylabel('Value', fontweight='bold')
                ax.set_title('STL Decomposition (Trend + Seasonal)', 
                           fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Decomposition not available\n(insufficient data)', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
    
    def _plot_spectral_analysis(self, ax1, ax2):
        """Plot spectral analysis results"""
        if 'spectral' in self.results:
            spec = self.results['spectral']
            
            # Periodogram
            freqs = spec['periodogram']['frequencies']
            psd = spec['periodogram']['psd']
            
            ax1.semilogy(freqs[1:], psd[1:], linewidth=1.5)  # Skip DC component
            ax1.set_xlabel('Frequency', fontweight='bold')
            ax1.set_ylabel('Power Spectral Density (log)', fontweight='bold')
            ax1.set_title('Periodogram (Frequency Domain)', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Mark dominant frequencies
            if spec['dominant_frequencies']:
                for dom in spec['dominant_frequencies'][:3]:
                    ax1.axvline(dom['frequency'], color='red', 
                              linestyle='--', alpha=0.5)
            
            # Dominant frequencies table
            ax2.axis('off')
            if spec['dominant_frequencies']:
                table_data = []
                for i, dom in enumerate(spec['dominant_frequencies'][:5], 1):
                    table_data.append([
                        f"{i}",
                        f"{dom['period']:.1f}",
                        f"{dom['power']:.2e}"
                    ])
                
                table = ax2.table(cellText=table_data,
                                colLabels=['#', 'Period', 'Power'],
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.2, 0.4, 0.4])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                
                ax2.set_title('Top Cycles', fontsize=11, fontweight='bold')
    
    def _plot_correlation_suite(self, ax_acf, ax_pacf, ax_ljung):
        """Plot ACF, PACF, and Ljung-Box test"""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        # ACF
        plot_acf(self.analyzer.prices, lags=min(24, self.analyzer.n//2), 
                ax=ax_acf, alpha=0.05)
        ax_acf.set_title('Autocorrelation (ACF)', fontweight='bold')
        ax_acf.grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(self.analyzer.prices, lags=min(24, self.analyzer.n//2), 
                 ax=ax_pacf, alpha=0.05, method='ywm')
        ax_pacf.set_title('Partial Autocorr (PACF)', fontweight='bold')
        ax_pacf.grid(True, alpha=0.3)
        
        # Ljung-Box p-values
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(self.analyzer.prices, lags=min(20, self.analyzer.n//3), 
                           return_df=True)
        
        ax_ljung.plot(lb.index, lb['lb_pvalue'], 'o-', linewidth=2)
        ax_ljung.axhline(0.05, color='red', linestyle='--', label='5% threshold')
        ax_ljung.set_xlabel('Lag', fontweight='bold')
        ax_ljung.set_ylabel('p-value', fontweight='bold')
        ax_ljung.set_title('Ljung-Box Test', fontweight='bold')
        ax_ljung.legend()
        ax_ljung.grid(True, alpha=0.3)
    
    def _plot_volatility_analysis(self, ax1, ax2):
        """Plot volatility clustering and rolling volatility"""
        dates = self.analyzer.dates
        returns = self.analyzer.simple_returns
        
        # Returns with GARCH-like clusters
        ax1_twin = ax1.twinx()
        ax1.bar(dates[1:], returns, alpha=0.3, color='steelblue', width=20)
        
        # Rolling volatility
        rolling_vol = pd.Series(returns).rolling(12).std().values
        ax1_twin.plot(dates[1:], rolling_vol, color='red', 
                     linewidth=2, label='Rolling Vol (12)')
        
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Returns', fontweight='bold')
        ax1_twin.set_ylabel('Rolling Volatility', fontweight='bold')
        ax1.set_title('Volatility Clustering', fontsize=11, fontweight='bold')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Volatility regime classification
        if 'volatility' in self.results:
            vol = self.results['volatility']
            
            # Create regime plot
            regimes = ['Low', 'Normal', 'High']
            colors = ['green', 'gray', 'red']
            
            # Simplified regime visualization
            ax2.text(0.5, 0.7, f"Current Regime:", ha='center', 
                    fontsize=12, fontweight='bold')
            ax2.text(0.5, 0.4, vol['regime']['classification'].upper(), 
                    ha='center', fontsize=20, fontweight='bold',
                    color=colors[['low', 'normal', 'high'].index(vol['regime']['classification'])])
            ax2.text(0.5, 0.2, f"Ratio: {vol['regime']['ratio']:.2f}x avg", 
                    ha='center', fontsize=10)
            ax2.axis('off')
            ax2.set_title('Volatility Regime', fontsize=11, fontweight='bold')
    
    def _plot_risk_analysis(self, ax1, ax2):
        """Plot risk metrics including drawdown"""
        # Drawdown chart
        cummax = np.maximum.accumulate(self.analyzer.prices)
        drawdown = (self.analyzer.prices - cummax) / cummax * 100
        
        ax1.fill_between(self.analyzer.dates, drawdown, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        ax1.plot(self.analyzer.dates, drawdown, color='darkred', linewidth=2)
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Drawdown (%)', fontweight='bold')
        ax1.set_title('Drawdown from Peak', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Risk metrics table
        ax2.axis('off')
        
        returns = self.analyzer.simple_returns
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        max_dd = np.min(drawdown)
        
        sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(12)
        
        risk_data = [
            ['VaR 95%', f'{var_95:.2f}%'],
            ['VaR 99%', f'{var_99:.2f}%'],
            ['Max DD', f'{max_dd:.2f}%'],
            ['Sharpe', f'{sharpe:.3f}'],
        ]
        
        table = ax2.table(cellText=risk_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)
        
        ax2.set_title('Risk Metrics', fontsize=11, fontweight='bold')
    
    def _plot_nonlinear_analysis(self, ax1, ax2, ax3):
        """Plot non-linear dynamics results"""
        if 'nonlinear' in self.results:
            nl = self.results['nonlinear']
            
            # Hurst exponent interpretation
            ax1.axis('off')
            hurst = nl['hurst_exponent']['rs_hurst']
            interp = nl['hurst_exponent']['interpretation']
            
            # Color code
            if hurst > 0.6:
                color = 'green'
                text = 'TRENDING'
            elif hurst < 0.4:
                color = 'blue'
                text = 'MEAN-REVERTING'
            else:
                color = 'gray'
                text = 'RANDOM WALK'
            
            ax1.text(0.5, 0.6, f'Hurst: {hurst:.3f}', ha='center', 
                    fontsize=16, fontweight='bold')
            ax1.text(0.5, 0.3, text, ha='center', fontsize=14, 
                    fontweight='bold', color=color)
            ax1.set_title('Long Memory', fontsize=11, fontweight='bold')
            
            # Phase space reconstruction (2D)
            embedded = self._embed_series(self.analyzer.prices, dim=2, lag=1)
            ax2.plot(embedded[:, 0], embedded[:, 1], alpha=0.6, linewidth=0.5)
            ax2.scatter(embedded[:, 0], embedded[:, 1], c=range(len(embedded)), 
                       cmap='viridis', s=10, alpha=0.5)
            ax2.set_xlabel('x(t)', fontweight='bold')
            ax2.set_ylabel('x(t+1)', fontweight='bold')
            ax2.set_title('Phase Space', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Lyapunov/DFA results
            ax3.axis('off')
            results_text = "Non-linear Metrics:\n\n"
            
            if nl['lyapunov_exponent']:
                results_text += f"Lyapunov: {nl['lyapunov_exponent']:.6f}\n"
            if nl['dfa_alpha']:
                results_text += f"DFA α: {nl['dfa_alpha']:.3f}\n"
            if nl['correlation_dimension']:
                results_text += f"Corr Dim: {nl['correlation_dimension']:.2f}\n"
            
            ax3.text(0.1, 0.5, results_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
            ax3.set_title('Chaos Indicators', fontsize=11, fontweight='bold')
    
    def _embed_series(self, ts, dim=2, lag=1):
        """Time delay embedding"""
        n = len(ts) - (dim-1) * lag
        embedded = np.zeros((n, dim))
        for i in range(dim):
            embedded[:, i] = ts[i*lag:i*lag+n]
        return embedded
    
    def _plot_information_and_breaks(self, ax_info, ax_breaks):
        """Plot information theory and structural breaks"""
        # Information metrics
        if 'information' in self.results:
            info = self.results['information']
            ax_info.axis('off')
            
            info_text = "INFORMATION THEORY\n" + "="*40 + "\n\n"
            
            if 'sample_entropy' in info and not np.isnan(info['sample_entropy']):
                info_text += f"Sample Entropy: {info['sample_entropy']:.3f}\n"
            
            if 'lz_complexity' in info:
                info_text += f"LZ Complexity: {info['lz_complexity']:.3f}\n"
            
            if 'predictability' in info and info['predictability']:
                pred = info['predictability']
                info_text += f"\nPredictability: {pred['interpretation'].upper()}\n"
                info_text += f"AR(1) R²: {pred['ar1_r2']:.3f}\n"
            
            ax_info.text(0.05, 0.5, info_text, fontsize=10, 
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax_info.set_title('Complexity Analysis', fontsize=11, fontweight='bold')
        
        # Structural breaks
        if 'structural_breaks' in self.results:
            breaks = self.results['structural_breaks']
            
            if 'cusum' in breaks and breaks['cusum']['breaks_detected']:
                cusum_vals = breaks['cusum']['cusum_values']
                critical = breaks['cusum']['critical_value']
                
                ax_breaks.plot(cusum_vals, linewidth=2, label='CUSUM')
                ax_breaks.axhline(critical, color='red', linestyle='--', 
                                label=f'Critical ({critical:.1f})')
                ax_breaks.axhline(-critical, color='red', linestyle='--')
                ax_breaks.fill_between(range(len(cusum_vals)), -critical, critical, 
                                      alpha=0.2, color='green')
                ax_breaks.set_xlabel('Time', fontweight='bold')
                ax_breaks.set_ylabel('CUSUM', fontweight='bold')
                ax_breaks.set_title('CUSUM Test (Breaks)', fontsize=11, fontweight='bold')
                ax_breaks.legend(fontsize=8)
                ax_breaks.grid(True, alpha=0.3)
            else:
                ax_breaks.text(0.5, 0.5, 'No significant\nstructural breaks', 
                             ha='center', va='center', fontsize=12)
                ax_breaks.axis('off')
    
    def save_dashboard(self, filename='ultimate_analysis.png'):
        """Save dashboard to file"""
        fig = self.create_master_dashboard()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"✓ Dashboard saved to {filename}")
        return fig
    
    def generate_pdf_report(self, filename='ultimate_analysis_report.pdf'):
        """Generate comprehensive PDF report"""
        with PdfPages(filename) as pdf:
            # Page 1: Master dashboard
            fig = self.create_master_dashboard()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Detailed distribution analysis
            fig2 = self._create_distribution_page()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            # Page 3: ARIMA and model selection
            fig3 = self._create_arima_page()
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'Ultimate Time Series Analysis Report'
            d['Author'] = 'AgriProfit Analytics'
            d['Subject'] = 'Comprehensive Time Series Analysis'
            d['Keywords'] = 'Time Series, Statistics, Forecasting'
        
        print(f"✓ PDF report saved to {filename}")
    
    def _create_distribution_page(self):
        """Create detailed distribution analysis page"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detailed Distribution Analysis', fontsize=16, fontweight='bold')
        
        prices = self.analyzer.prices
        
        # Histogram with multiple distributions
        ax = axes[0, 0]
        ax.hist(prices, bins=30, alpha=0.5, density=True, label='Data')
        
        from scipy import stats
        mu, sigma = np.mean(prices), np.std(prices)
        x = np.linspace(prices.min(), prices.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Price')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Quantile-quantile plots for multiple distributions
        ax = axes[0, 1]
        stats.probplot(prices, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal)')
        ax.grid(True, alpha=0.3)
        
        # Empirical CDF
        ax = axes[1, 0]
        sorted_data = np.sort(prices)
        y = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        ax.plot(sorted_data, y, linewidth=2)
        ax.set_xlabel('Price')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Empirical CDF')
        ax.grid(True, alpha=0.3)
        
        # Tail analysis
        ax = axes[1, 1]
        returns = self.analyzer.simple_returns
        ax.hist(returns, bins=50, alpha=0.6, edgecolor='black')
        ax.axvline(np.percentile(returns, 5), color='red', 
                  linestyle='--', label='5th percentile')
        ax.axvline(np.percentile(returns, 95), color='red', 
                  linestyle='--', label='95th percentile')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution (Tails)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_arima_page(self):
        """Create ARIMA model selection page"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ARIMA Model Selection & Diagnostics', fontsize=16, fontweight='bold')
        
        if 'arima_identification' in self.results:
            arima = self.results['arima_identification']
            
            # Model comparison (AIC/BIC)
            ax = axes[0, 0]
            if arima['best_models']:
                models = arima['best_models'][:10]
                model_labels = [f"{m['order']}" for m in models]
                aics = [m['aic'] for m in models]
                bics = [m['bic'] for m in models]
                
                x = range(len(models))
                ax.plot(x, aics, 'o-', label='AIC', linewidth=2)
                ax.plot(x, bics, 's-', label='BIC', linewidth=2)
                ax.set_xticks(x)
                ax.set_xticklabels(model_labels, rotation=45, ha='right')
                ax.set_xlabel('ARIMA Model')
                ax.set_ylabel('Information Criterion')
                ax.set_title('Model Selection (Top 10)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # ACF for residual diagnostics
            ax = axes[0, 1]
            ax.text(0.5, 0.5, 'Best Model:\n' + str(models[0]['order']) if models else 'N/A',
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.text(0.5, 0.3, f"AIC: {models[0]['aic']:.2f}\nBIC: {models[0]['bic']:.2f}" if models else '',
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            ax.set_title('Selected Model')
            
            # Model suggestions
            ax = axes[1, 0]
            ax.axis('off')
            suggestions_text = "Model Suggestions:\n\n"
            for i, sug in enumerate(arima['suggestions'][:5], 1):
                suggestions_text += f"{i}. {sug}\n"
            ax.text(0.1, 0.5, suggestions_text, fontsize=10, 
                   verticalalignment='center', family='monospace')
            ax.set_title('Based on ACF/PACF')
            
            # Significant lags
            ax = axes[1, 1]
            sig_acf = arima['acf_significant_lags']
            sig_pacf = arima['pacf_significant_lags']
            
            ax.scatter(sig_acf, [1]*len(sig_acf), s=100, alpha=0.6, 
                      label=f'ACF ({len(sig_acf)} lags)')
            ax.scatter(sig_pacf, [0]*len(sig_pacf), s=100, alpha=0.6,
                      label=f'PACF ({len(sig_pacf)} lags)')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['PACF', 'ACF'])
            ax.set_xlabel('Lag')
            ax.set_title('Significant Lags')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# =====================================================================
# INTEGRATION WITH AGRIPROFIT STREAMLIT APP
# =====================================================================

def integrate_with_agriprofit(prices_df):
    """
    Complete integration function for AgriProfit system
    
    Args:
        prices_df: DataFrame with 'date' and 'price' columns
    
    Returns:
        dict with all results and figure objects
    """
    
    print("="*80)
    print("ULTIMATE TIME SERIES ANALYSIS - AgriProfit Integration")
    print("="*80)
    
    # Step 1: Create analyzer
    analyzer = UltimateTimeSeriesAnalyzer(prices_df, frequency='M')
    
    # Step 2: Run complete analysis
    results = analyzer.run_ultimate_analysis()
    
    # Step 3: Create visualizer
    visualizer = UltimateVisualizer(analyzer)
    
    # Step 4: Generate outputs
    print("\nGenerating visualizations...")
    
    # Master dashboard
    fig_dashboard = visualizer.create_master_dashboard()
    
    # Save outputs
    visualizer.save_dashboard('agriprofit_ultimate_analysis.png')
    visualizer.generate_pdf_report('agriprofit_ultimate_report.pdf')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  ✓ agriprofit_ultimate_analysis.png (Dashboard)")
    print("  ✓ agriprofit_ultimate_report.pdf (Full Report)")
    
    return {
        'analyzer': analyzer,
        'results': results,
        'visualizer': visualizer,
        'figure': fig_dashboard
    }


# =====================================================================
# STREAMLIT APP INTEGRATION
# =====================================================================

def create_streamlit_tab(prices_df):
    """
    Function to add Ultimate Analysis tab to AgriProfit Streamlit app
    
    Add this to your ui/app_streamlit.py:
    
    ```python
    from ultimate_visualizer import create_streamlit_tab
    
    # In your tabs definition:
    tabs = st.tabs([..., "Ultimate Analysis"])
    
    with tabs[-1]:
        create_streamlit_tab(prices)
    ```
    """
    import streamlit as st
    
    st.header("🔬 Ultimate Time Series Analysis")
    st.caption("Professional-grade deep analysis with 15+ statistical methods")
    
    with st.spinner("Running comprehensive analysis..."):
        # Create analyzer
        analyzer = UltimateTimeSeriesAnalyzer(prices_df, frequency='M')
        
        # Run analysis
        results = analyzer.run_ultimate_analysis()
        
        # Create visualizer
        visualizer = UltimateVisualizer(analyzer)
    
    # Show dashboard
    st.subheader("Analysis Dashboard")
    fig = visualizer.create_master_dashboard()
    st.pyplot(fig)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Save PNG
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="📥 Download Dashboard (PNG)",
            data=buf,
            file_name="ultimate_analysis.png",
            mime="image/png"
        )
    
    with col2:
        # Generate PDF
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        pdf_buf.seek(0)
        
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_buf,
            file_name="ultimate_analysis.pdf",
            mime="application/pdf"
        )
    
    # Expandable sections for detailed results
    with st.expander("📊 Distribution Analysis"):
        if 'distribution_advanced' in results:
            dist = results['distribution_advanced']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Shannon Entropy", f"{dist['information']['shannon_entropy']:.3f}")
            col2.metric("Sample Entropy", f"{dist['information']['sample_entropy']:.3f}")
            col3.metric("Gini Coefficient", f"{dist['moments']['gini']:.3f}")
            col4.metric("CV", f"{dist['moments']['cv']*100:.1f}%")
            
            st.write("**Shape Tests:**")
            st.json({
                'Jarque-Bera': f"stat={dist['shape_tests']['jarque_bera'][0]:.3f}, p={dist['shape_tests']['jarque_bera'][1]:.4f}",
                'Skewness': f"{dist['moments']['skewness']:.3f}",
                'Kurtosis': f"{dist['moments']['kurtosis']:.3f}"
            })
    
    with st.expander("🌊 Spectral Analysis"):
        if 'spectral' in results:
            spec = results['spectral']
            
            st.write("**Dominant Frequencies (Top 5 Cycles):**")
            if spec['dominant_frequencies']:
                df_spec = pd.DataFrame(spec['dominant_frequencies'][:5])
                df_spec['period'] = df_spec['period'].round(2)
                df_spec['frequency'] = df_spec['frequency'].round(4)
                st.dataframe(df_spec)
            
            col1, col2 = st.columns(2)
            col1.metric("Spectral Entropy", f"{spec['spectral_entropy']:.3f}")
            col2.metric("Low Freq Power", f"{spec['band_power']['low_freq']*100:.1f}%")
    
    with st.expander("🔄 Non-Linear Dynamics"):
        if 'nonlinear' in results:
            nl = results['nonlinear']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Hurst Exponent", f"{nl['hurst_exponent']['rs_hurst']:.3f}",
                       delta=nl['hurst_exponent']['interpretation'])
            if nl['lyapunov_exponent']:
                col2.metric("Lyapunov Exp", f"{nl['lyapunov_exponent']:.6f}")
            if nl['dfa_alpha']:
                col3.metric("DFA Alpha", f"{nl['dfa_alpha']:.3f}")
            
            st.info(f"**Interpretation:** {nl['hurst_exponent']['interpretation'].upper()} behavior detected")
    
    with st.expander("📈 Stationarity Tests"):
        if 'stationarity_advanced' in results:
            stat = results['stationarity_advanced']
            
            st.write(f"**Conclusion:** {stat['conclusion'].replace('_', ' ').title()}")
            
            if stat['tests']['adf_multi']:
                st.write("**ADF Tests:**")
                df_adf = pd.DataFrame(stat['tests']['adf_multi'])
                st.dataframe(df_adf)
            
            if stat['tests']['variance_ratio']:
                st.write("**Variance Ratio Tests:**")
                df_vr = pd.DataFrame(stat['tests']['variance_ratio'])
                st.dataframe(df_vr)
    
    with st.expander("🎯 ARIMA Model Selection"):
        if 'arima_identification' in results:
            arima = results['arima_identification']
            
            if arima['best_models']:
                st.write("**Top 10 Models by AIC:**")
                df_models = pd.DataFrame(arima['best_models'])
                df_models['order'] = df_models['order'].astype(str)
                st.dataframe(df_models)
                
                st.success(f"**Best Model:** ARIMA{arima['best_models'][0]['order']} "
                          f"(AIC={arima['best_models'][0]['aic']:.2f})")
            
            if arima['suggestions']:
                st.write("**Suggestions based on ACF/PACF:**")
                for sug in arima['suggestions']:
                    st.write(f"- {sug}")
    
    with st.expander("⚡ Structural Breaks"):
        if 'structural_breaks' in results:
            breaks = results['structural_breaks']
            
            if breaks['cusum']['breaks_detected']:
                st.warning(f"CUSUM test detected structural instability at "
                          f"{len(breaks['cusum']['break_indices'])} points")
            
            if breaks['chow_tests']:
                st.write("**Significant Chow Tests:**")
                df_chow = pd.DataFrame([b for b in breaks['chow_tests'] if b['significant']])
                if len(df_chow) > 0:
                    st.dataframe(df_chow)
                else:
                    st.info("No significant breaks detected by Chow test")
    
    with st.expander("🎲 Extreme Value Analysis"):
        if 'extreme_value' in results and 'error' not in results['extreme_value']:
            evt = results['extreme_value']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Extreme VaR 95%", f"{evt['extreme_var']['var_95']:.2f}%")
            col2.metric("Extreme VaR 99%", f"{evt['extreme_var']['var_99']:.2f}%")
            col3.metric("Extreme VaR 99.9%", f"{evt['extreme_var']['var_999']:.2f}%")
            
            st.write("**GPD Parameters:**")
            st.json(evt['gpd_parameters'])
    
    with st.expander("🧠 Information Theory & Predictability"):
        if 'information' in results:
            info = results['information']
            
            if info['predictability']:
                pred = info['predictability']
                
                st.metric("Predictability", pred['interpretation'].upper(),
                         delta=f"R² = {pred['ar1_r2']:.3f}")
                
                if pred['interpretation'] == 'high':
                    st.success("✅ Series shows high predictability - forecasting likely effective")
                elif pred['interpretation'] == 'low':
                    st.warning("⚠️ Series shows low predictability - forecasting challenging")
            
            col1, col2 = st.columns(2)
            if not np.isnan(info['sample_entropy']):
                col1.metric("Sample Entropy", f"{info['sample_entropy']:.3f}")
            col2.metric("LZ Complexity", f"{info['lz_complexity']:.3f}")


# =====================================================================
# EXAMPLE USAGE & TESTING
# =====================================================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=60, freq='MS')
    
    t = np.arange(60)
    trend = 15 + 0.15 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    cycle = 2 * np.sin(2 * np.pi * t / 36)
    shocks = np.zeros(60)
    shocks[15] = -4
    shocks[40] = -3
    noise = np.random.normal(0, 1, 60)
    
    prices = np.maximum(8, trend + seasonal + cycle + shocks + noise)
    
    df = pd.DataFrame({'date': dates, 'price': prices})
    
    # Run complete integration
    output = integrate_with_agriprofit(df)
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE!")
    print("="*80)
    print("\nAccess results:")
    print("  - output['results']: All analysis results")
    print("  - output['analyzer']: Analyzer object")
    print("  - output['visualizer']: Visualizer object")
    print("  - output['figure']: Main dashboard figure")
    print("\nFiles created:")
    print("  - agriprofit_ultimate_analysis.png")
    print("  - agriprofit_ultimate_report.pdf")
    print("="*80)