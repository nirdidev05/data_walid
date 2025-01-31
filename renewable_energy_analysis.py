import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class RenewableEnergyData:
    """Data class to store renewable energy information with expanded metrics."""
    country: str
    year: int
    solar_capacity: float
    wind_capacity: float
    hydro_capacity: float
    total_renewable: float
    carbon_offset: float  # Estimated CO2 offset in metric tons
    investment_usd: float  # Investment in millions USD
    efficiency_ratio: float  # Energy output vs capacity
    implementation_cost: float  # Cost per MW installed

class DataScraper:
    """Handle web scraping operations for renewable energy data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def fetch_irena_data(self, country: str, year: int) -> Dict:
        """
        Fetch data from IRENA (International Renewable Energy Agency).
        Note: This is a template - you'd need to adjust URLs and parsing logic
        for the actual IRENA website structure.
        """
        base_url = f"https://www.irena.org/Statistics/View-Data-by-Topic/Capacity-and-Generation/Country-Rankings"
        try:
            response = self.session.get(f"{base_url}?country={country}&year={year}", 
                                      headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse the data (example parsing logic - adjust based on actual website structure)
            data = {
                'solar_capacity': self._parse_value(soup.find('div', {'id': 'solar-capacity'})),
                'wind_capacity': self._parse_value(soup.find('div', {'id': 'wind-capacity'})),
                'hydro_capacity': self._parse_value(soup.find('div', {'id': 'hydro-capacity'}))
            }
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching IRENA data: {str(e)}")
            return {}

    def fetch_worldbank_data(self, country: str, year: int) -> Dict:
        """
        Fetch investment and implementation cost data from World Bank API.
        """
        api_url = f"https://api.worldbank.org/v2/country/{country}/indicator/EG.REN.INVEST"
        try:
            response = self.session.get(api_url, params={'date': year, 'format': 'json'})
            response.raise_for_status()
            data = response.json()
            # Process World Bank API response (example)
            return {
                'investment_usd': self._extract_worldbank_value(data, 'investment'),
                'implementation_cost': self._extract_worldbank_value(data, 'cost')
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching World Bank data: {str(e)}")
            return {}

    @staticmethod
    def _parse_value(element) -> float:
        """Parse numeric value from HTML element."""
        return float(element.text.strip().replace(',', '')) if element else 0.0

    @staticmethod
    def _extract_worldbank_value(data: Dict, key: str) -> float:
        """Extract and process values from World Bank API response."""
        try:
            return float(data[1][0]['value'])
        except (KeyError, IndexError, TypeError):
            return 0.0

class RenewableEnergyAnalyzer:
    """Enhanced analyzer with advanced features."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.data: List[RenewableEnergyData] = []
        self.scraper = DataScraper()

    def fetch_data(self, start_year: int, end_year: int) -> None:
        """Fetch data with parallel processing for faster data collection."""
        cache_file = os.path.join(self.cache_dir, f"energy_data_{start_year}_{end_year}.json")
        
        if os.path.exists(cache_file):
            self.logger.info("Loading data from cache...")
            with open(cache_file, 'r') as f:
                self.data = [RenewableEnergyData(**item) for item in json.load(f)]
            return

        self.logger.info("Fetching fresh data from multiple sources...")
        countries = ['USA', 'China', 'Germany', 'India', 'Brazil', 'Spain', 'Australia']
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for year in range(start_year, end_year + 1):
                for country in countries:
                    futures.append(
                        executor.submit(self._fetch_country_data, country, year)
                    )
            
            for future in futures:
                result = future.result()
                if result:
                    self.data.append(result)

        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump([vars(item) for item in self.data], f)

    def _fetch_country_data(self, country: str, year: int) -> Optional[RenewableEnergyData]:
        """Fetch and combine data from multiple sources for a single country-year pair."""
        try:
            irena_data = self.scraper.fetch_irena_data(country, year)
            worldbank_data = self.scraper.fetch_worldbank_data(country, year)
            
            if not irena_data or not worldbank_data:
                return None

            total_renewable = sum(irena_data.values())
            carbon_offset = self._calculate_carbon_offset(total_renewable)
            efficiency_ratio = self._calculate_efficiency_ratio(total_renewable, worldbank_data['implementation_cost'])

            return RenewableEnergyData(
                country=country,
                year=year,
                solar_capacity=irena_data['solar_capacity'],
                wind_capacity=irena_data['wind_capacity'],
                hydro_capacity=irena_data['hydro_capacity'],
                total_renewable=total_renewable,
                carbon_offset=carbon_offset,
                investment_usd=worldbank_data['investment_usd'],
                efficiency_ratio=efficiency_ratio,
                implementation_cost=worldbank_data['implementation_cost']
            )
        except Exception as e:
            self.logger.error(f"Error fetching data for {country} {year}: {str(e)}")
            return None

    def analyze_trends(self) -> Tuple[pd.DataFrame, Dict]:
        """Perform advanced trend analysis including statistical tests and clustering."""
        df = pd.DataFrame([vars(item) for item in self.data])
        
        # Basic trend analysis
        df_grouped = df.groupby(['country', 'year']).agg({
            'total_renewable': 'sum',
            'solar_capacity': 'sum',
            'wind_capacity': 'sum',
            'hydro_capacity': 'sum',
            'carbon_offset': 'sum',
            'investment_usd': 'sum',
            'efficiency_ratio': 'mean',
            'implementation_cost': 'mean'
        }).reset_index()

        # Calculate advanced metrics
        analysis_results = {}
        
        # Perform clustering analysis
        scaler = StandardScaler()
        features = ['total_renewable', 'efficiency_ratio', 'implementation_cost']
        scaled_features = scaler.fit_transform(df_grouped[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_grouped['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Statistical tests
        analysis_results['correlation_matrix'] = df_grouped[features].corr()
        
        # Trend significance tests
        for country in df_grouped['country'].unique():
            country_data = df_grouped[df_grouped['country'] == country]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                country_data['year'],
                country_data['total_renewable']
            )
            analysis_results[f'{country}_trend_test'] = {
                'slope': slope,
                'p_value': p_value,
                'r_squared': r_value**2
            }

        return df_grouped, analysis_results


    def generate_visualizations(self, df: pd.DataFrame, analysis_results: Dict) -> None:
        """Generate advanced visualizations."""
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Interactive 3D Scatter Plot
        fig_3d = px.scatter_3d(df, 
                              x='total_renewable', 
                              y='efficiency_ratio', 
                              z='implementation_cost',
                              color='country',
                              size='investment_usd',
                              animation_frame='year',
                              title='Renewable Energy Metrics - 3D View')
        fig_3d.write_html('visualizations/3d_analysis.html')

        # 2. Advanced Heatmap with Multiple Metrics
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=analysis_results['correlation_matrix'],
            x=analysis_results['correlation_matrix'].columns,
            y=analysis_results['correlation_matrix'].columns,
            colorscale='Viridis'))
        fig_heatmap.update_layout(title='Correlation Matrix of Key Metrics')
        fig_heatmap.write_html('visualizations/correlation_heatmap.html')

        # 3. Sophisticated Time Series Dashboard
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Renewable Capacity', 'Energy Mix Evolution',
                          'Investment Trends', 'Efficiency Metrics')
        )

        # Add various plots to dashboard
        colors = px.colors.qualitative.Set3
        for i, country in enumerate(df['country'].unique()):
            country_data = df[df['country'] == country].sort_values('year')
            color = colors[i % len(colors)]
            
            # Total capacity trend
            fig_dashboard.add_trace(
                go.Scatter(x=country_data['year'], 
                          y=country_data['total_renewable'],
                          name=f"{country} - Total",
                          mode='lines+markers',
                          line=dict(color=color)),
                row=1, col=1
            )
            
            # Energy mix - Using Scatter with fill instead of Area
            fig_dashboard.add_trace(
                go.Scatter(x=country_data['year'],
                          y=country_data['solar_capacity'],
                          name=f"{country} - Solar",
                          fill='tonexty',
                          line=dict(color=color)),
                row=1, col=2
            )

            # Investment scatter
            fig_dashboard.add_trace(
                go.Scatter(x=country_data['implementation_cost'],
                          y=country_data['investment_usd'],
                          name=country,
                          mode='markers',
                          marker=dict(size=10, color=color)),
                row=2, col=1
            )

            # Efficiency metrics
            fig_dashboard.add_trace(
                go.Bar(x=country_data['year'],
                      y=country_data['efficiency_ratio'],
                      name=f"{country} - Efficiency",
                      marker_color=color),
                row=2, col=2
            )

        fig_dashboard.update_layout(
            height=1000, 
            width=1200, 
            title_text="Renewable Energy Dashboard",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        # Update layout for better readability
        fig_dashboard.update_xaxes(title_text="Year", row=1, col=1)
        fig_dashboard.update_xaxes(title_text="Year", row=1, col=2)
        fig_dashboard.update_xaxes(title_text="Implementation Cost", row=2, col=1)
        fig_dashboard.update_xaxes(title_text="Year", row=2, col=2)
        
        fig_dashboard.update_yaxes(title_text="Total Renewable (MW)", row=1, col=1)
        fig_dashboard.update_yaxes(title_text="Capacity (MW)", row=1, col=2)
        fig_dashboard.update_yaxes(title_text="Investment (USD M)", row=2, col=1)
        fig_dashboard.update_yaxes(title_text="Efficiency Ratio", row=2, col=2)
        
        fig_dashboard.write_html('visualizations/dashboard.html')

    def generate_report(self, df: pd.DataFrame, analysis_results: Dict) -> str:
        """Generate comprehensive analysis report with advanced insights."""
        report = []
        report.append("# Comprehensive Renewable Energy Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary")
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        total_capacity = latest_data['total_renewable'].sum()
        total_investment = latest_data['investment_usd'].sum()
        avg_efficiency = latest_data['efficiency_ratio'].mean()
        
        report.append(f"### Key Findings")
        report.append(f"- Total renewable capacity in {latest_year}: {total_capacity:,.2f} MW")
        report.append(f"- Total investment: ${total_investment:,.2f} million")
        report.append(f"- Average efficiency ratio: {avg_efficiency:.2f}")

        # Trend Analysis
        report.append("\n## Trend Analysis")
        for country in df['country'].unique():
            trend_test = analysis_results[f'{country}_trend_test']
            report.append(f"\n### {country}")
            report.append(f"- Annual growth rate: {trend_test['slope']:,.2f} MW/year")
            report.append(f"- Trend significance (p-value): {trend_test['p_value']:.4f}")
            report.append(f"- Model fit (R²): {trend_test['r_squared']:.4f}")

        # Investment Analysis
        report.append("\n## Investment Efficiency Analysis")
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            roi = (country_data['total_renewable'] / country_data['investment_usd']).mean()
            report.append(f"\n### {country}")
            report.append(f"- Return on Investment: {roi:.2f} MW per million USD")
            report.append(f"- Average implementation cost: ${country_data['implementation_cost'].mean():,.2f}/MW")

        # Environmental Impact
        report.append("\n## Environmental Impact Assessment")
        total_carbon_offset = df['carbon_offset'].sum()
        report.append(f"\nTotal estimated carbon offset: {total_carbon_offset:,.2f} metric tons CO2")
        
        # Recommendations
        report.append("\n## Recommendations")
        efficient_countries = df.nlargest(3, 'efficiency_ratio')['country'].unique()
        report.append("\nBest practices based on top performers:")
        for country in efficient_countries:
            country_data = df[df['country'] == country]
            report.append(f"\n### Lessons from {country}")
            report.append(f"- Efficiency ratio: {country_data['efficiency_ratio'].mean():.2f}")
            report.append(f"- Investment strategy: ${country_data['investment_usd'].mean():,.2f} million/year")

        return "\n".join(report)

    @staticmethod
    def _calculate_carbon_offset(renewable_capacity: float) -> float:
        """Calculate estimated carbon offset based on renewable capacity."""
        # Simplified calculation - in reality, would need more complex modeling
        return renewable_capacity * 0.7  # Assume 0.7 metric tons CO2 offset per MW

    @staticmethod
    def _calculate_efficiency_ratio(total_renewable: float, implementation_cost: float) -> float:
        """Calculate efficiency ratio based on output vs cost."""
        if implementation_cost == 0:
            return 0.0
        return (total_renewable / implementation_cost) * 100

def main():
    """Main function to run the enhanced analysis."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('renewable_energy_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize analyzer
    analyzer = RenewableEnergyAnalyzer()
    
    try:
        # Create necessary directories
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Set analysis parameters
        start_year = 2015
        end_year = 2023
        
        # Execute analysis pipeline
        logging.info("Starting renewable energy analysis...")
        
        # Step 1: Data Collection
        logging.info("Fetching data...")
        analyzer.fetch_data(start_year, end_year)
        
        # Step 2: Analysis
        logging.info("Analyzing trends...")
        df_analysis, analysis_results = analyzer.analyze_trends()
        
        # Step 3: Visualizations
        logging.info("Generating visualizations...")
        analyzer.generate_visualizations(df_analysis, analysis_results)
        
        # Step 4: Report Generation
        logging.info("Generating report...")
        report = analyzer.generate_report(df_analysis, analysis_results)
        
        # Save report
        report_file = f'reports/renewable_energy_report_{datetime.now().strftime("%Y%m%d")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
            
        # Export analyzed data
        df_analysis.to_csv('data_cache/analyzed_data.csv', index=False)
        
        logging.info("Analysis completed successfully!")
        logging.info(f"Report saved to: {report_file}")
        logging.info("Visualizations saved to: ./visualizations/")
        
    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()