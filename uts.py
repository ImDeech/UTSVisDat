"""
Exercise Data Visualization Dashboard - Complete Implementation
Menggunakan teknik dari berbagai Kaggle notebooks:
- HR Analytics: Statistical analysis & correlation
- NBA 2K20: Distribution & comparison charts
- Netflix: Interactive plotly visualizations
- TED Talks: Time series & trend analysis
- Penguins: Seaborn advanced styling
- Cryptocurrency: Predictive insights
- YouTube Music: Multi-metric analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)

# ==================== DATA GENERATION ====================
def generate_exercise_data(n_samples=200):
    """Generate realistic exercise data"""
    np.random.seed(42)
    
    exercises = ['Running', 'Cycling', 'Swimming', 'Weight Training', 
                 'Yoga', 'HIIT', 'Walking', 'Elliptical', 'Boxing']
    
    data = []
    for i in range(n_samples):
        exercise = np.random.choice(exercises)
        
        # Realistic parameters
        params = {
            'Running': {'dur': (30, 90), 'bpm': (140, 180), 'cal_rate': (6, 10)},
            'Cycling': {'dur': (40, 100), 'bpm': (130, 165), 'cal_rate': (5, 8)},
            'Swimming': {'dur': (30, 80), 'bpm': (135, 170), 'cal_rate': (7, 11)},
            'Weight Training': {'dur': (45, 90), 'bpm': (110, 140), 'cal_rate': (3, 6)},
            'Yoga': {'dur': (50, 90), 'bpm': (85, 115), 'cal_rate': (2, 4)},
            'HIIT': {'dur': (20, 45), 'bpm': (160, 190), 'cal_rate': (10, 15)},
            'Walking': {'dur': (40, 90), 'bpm': (100, 130), 'cal_rate': (2, 4)},
            'Elliptical': {'dur': (30, 60), 'bpm': (125, 155), 'cal_rate': (5, 8)},
            'Boxing': {'dur': (30, 60), 'bpm': (150, 180), 'cal_rate': (8, 12)}
        }
        
        p = params[exercise]
        duration = np.random.uniform(*p['dur'])
        bpm = np.random.uniform(*p['bpm'])
        cal_rate = np.random.uniform(*p['cal_rate'])
        calories = duration * cal_rate
        
        data.append({
            'Exercise': exercise,
            'Duration': round(duration, 1),
            'BPM': round(bpm),
            'Calories': round(calories),
            'Efficiency': round(calories/duration, 2)
        })
    
    return pd.DataFrame(data)

# ==================== DATA ANALYSIS ====================
def analyze_data(df):
    """Statistical analysis - from HR Analytics notebook"""
    print("="*70)
    print("EXERCISE DATA ANALYSIS REPORT")
    print("="*70)
    
    print("\n1. DATASET OVERVIEW")
    print(f"Total Records: {len(df)}")
    print(f"Features: {df.shape[1]}")
    print(f"\nExercise Types: {df['Exercise'].nunique()}")
    print(df['Exercise'].value_counts())
    
    print("\n2. DESCRIPTIVE STATISTICS")
    print(df.describe())
    
    print("\n3. CORRELATION ANALYSIS")
    numeric_cols = ['Duration', 'BPM', 'Calories', 'Efficiency']
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    print("\n4. KEY INSIGHTS")
    print(f"Highest Avg Calories: {df.groupby('Exercise')['Calories'].mean().idxmax()}")
    print(f"Most Efficient: {df.groupby('Exercise')['Efficiency'].mean().idxmax()}")
    print(f"Highest Avg BPM: {df.groupby('Exercise')['BPM'].mean().idxmax()}")
    
    return df

# ==================== VISUALIZATION 1: Overview Dashboard ====================
def create_overview_dashboard(df):
    """Create comprehensive overview - from Netflix notebook style"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Exercise Distribution', 'Average Calories by Exercise',
                       'BPM vs Calories', 'Duration Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'box'}]]
    )
    
    # 1. Pie Chart - Exercise Distribution
    exercise_counts = df['Exercise'].value_counts()
    fig.add_trace(
        go.Pie(labels=exercise_counts.index, values=exercise_counts.values,
               hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)),
        row=1, col=1
    )
    
    # 2. Bar Chart - Average Calories
    avg_cal = df.groupby('Exercise')['Calories'].mean().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(x=avg_cal.values, y=avg_cal.index, orientation='h',
               marker=dict(color=avg_cal.values, colorscale='Oranges')),
        row=1, col=2
    )
    
    # 3. Scatter - BPM vs Calories
    fig.add_trace(
        go.Scatter(x=df['BPM'], y=df['Calories'], mode='markers',
                  marker=dict(size=8, color=df['Duration'], colorscale='Viridis',
                            showscale=True, colorbar=dict(title="Duration")),
                  text=df['Exercise']),
        row=2, col=1
    )
    
    # 4. Box Plot - Duration Distribution
    for exercise in df['Exercise'].unique():
        exercise_data = df[df['Exercise'] == exercise]['Duration']
        fig.add_trace(
            go.Box(y=exercise_data, name=exercise, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, 
                     title_text="Exercise Analytics Dashboard - Overview",
                     title_font_size=20)
    fig.write_html("exercise_dashboard_overview.html")
    print("\n✅ Overview dashboard saved: exercise_dashboard_overview.html")
    return fig

# ==================== VISUALIZATION 2: Statistical Analysis ====================
def create_statistical_plots(df):
    """Deep statistical analysis - from Penguins notebook"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Analysis of Exercise Data', fontsize=18, fontweight='bold')
    
    # 1. Correlation Heatmap
    numeric_cols = ['Duration', 'BPM', 'Calories', 'Efficiency']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[0,0], cbar_kws={'label': 'Correlation'})
    axes[0,0].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 2. Distribution with KDE
    for col in ['Calories', 'BPM']:
        sns.histplot(data=df, x=col, kde=True, ax=axes[0,1], alpha=0.5, label=col)
    axes[0,1].set_title('Calories & BPM Distribution', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    
    # 3. Violin Plot - Duration by Exercise
    sns.violinplot(data=df, x='Exercise', y='Duration', ax=axes[1,0], palette='muted')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha='right')
    axes[1,0].set_title('Duration Distribution by Exercise Type', fontsize=14, fontweight='bold')
    
    # 4. Efficiency Comparison
    efficiency_avg = df.groupby('Exercise')['Efficiency'].mean().sort_values()
    efficiency_avg.plot(kind='barh', ax=axes[1,1], color='teal', alpha=0.7)
    axes[1,1].set_title('Average Efficiency (Cal/Min) by Exercise', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Calories per Minute')
    
    plt.tight_layout()
    plt.savefig('exercise_statistical_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Statistical plots saved: exercise_statistical_analysis.png")
    plt.show()

# ==================== VISUALIZATION 3: Interactive Plotly Dashboard ====================
def create_interactive_dashboard(df):
    """Interactive dashboard - from YouTube Music notebook style"""
    fig = go.Figure()
    
    # Create traces for each exercise type
    for exercise in df['Exercise'].unique():
        exercise_df = df[df['Exercise'] == exercise]
        fig.add_trace(go.Scatter(
            x=exercise_df['Duration'],
            y=exercise_df['Calories'],
            mode='markers',
            name=exercise,
            marker=dict(size=exercise_df['BPM']/10, opacity=0.6),
            text=[f"{exercise}<br>BPM: {bpm}<br>Cal: {cal}" 
                  for bpm, cal in zip(exercise_df['BPM'], exercise_df['Calories'])],
            hovertemplate='<b>%{text}</b><br>Duration: %{x} min<extra></extra>'
        ))
    
    fig.update_layout(
        title='Interactive Exercise Analysis: Duration vs Calories (Size = BPM)',
        xaxis_title='Duration (minutes)',
        yaxis_title='Calories Burned',
        hovermode='closest',
        height=600,
        template='plotly_white'
    )
    
    fig.write_html("exercise_interactive_scatter.html")
    print("✅ Interactive scatter saved: exercise_interactive_scatter.html")
    return fig

# ==================== VISUALIZATION 4: Comparative Analysis ====================
def create_comparative_analysis(df):
    """Multi-metric comparison - from NBA notebook"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparative Exercise Analysis', fontsize=18, fontweight='bold')
    
    # 1. Grouped Bar - Multiple Metrics
    metrics_df = df.groupby('Exercise').agg({
        'Calories': 'mean',
        'Duration': 'mean',
        'BPM': 'mean'
    }).reset_index()
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0,0].bar(x - width, metrics_df['Calories']/10, width, label='Calories/10', alpha=0.8)
    axes[0,0].bar(x, metrics_df['Duration'], width, label='Duration', alpha=0.8)
    axes[0,0].bar(x + width, metrics_df['BPM']/10, width, label='BPM/10', alpha=0.8)
    axes[0,0].set_xlabel('Exercise Type')
    axes[0,0].set_ylabel('Value')
    axes[0,0].set_title('Multi-Metric Comparison (Normalized)')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(metrics_df['Exercise'], rotation=45, ha='right')
    axes[0,0].legend()
    
    # 2. Box Plot - Calories Distribution
    df.boxplot(column='Calories', by='Exercise', ax=axes[0,1])
    axes[0,1].set_title('Calories Distribution by Exercise')
    axes[0,1].set_xlabel('Exercise Type')
    axes[0,1].set_ylabel('Calories')
    plt.sca(axes[0,1])
    plt.xticks(rotation=45, ha='right')
    
    # 3. Efficiency Ranking
    eff_stats = df.groupby('Exercise')['Efficiency'].agg(['mean', 'std']).sort_values('mean')
    eff_stats['mean'].plot(kind='barh', xerr=eff_stats['std'], ax=axes[1,0], 
                           color='coral', alpha=0.7, error_kw={'elinewidth': 2})
    axes[1,0].set_title('Exercise Efficiency with Standard Deviation')
    axes[1,0].set_xlabel('Efficiency (Cal/Min)')
    
    # 4. Count Plot
    exercise_counts = df['Exercise'].value_counts()
    axes[1,1].pie(exercise_counts.values, labels=exercise_counts.index, autopct='%1.1f%%',
                  startangle=90, colors=sns.color_palette('pastel'))
    axes[1,1].set_title('Exercise Type Distribution')
    
    plt.tight_layout()
    plt.savefig('exercise_comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Comparative analysis saved: exercise_comparative_analysis.png")
    plt.show()

# ==================== VISUALIZATION 5: Advanced Insights ====================
def create_advanced_insights(df):
    """Advanced analytics - from TED Talks & Crypto notebooks"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Efficiency Heatmap', 'BPM Range by Exercise',
                       'Calories Cumulative', 'Performance Score'),
        specs=[[{'type': 'heatmap'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Efficiency Heatmap
    pivot_eff = df.pivot_table(values='Efficiency', 
                                index='Exercise', 
                                aggfunc='mean').sort_values('Efficiency')
    fig.add_trace(
        go.Heatmap(z=[pivot_eff.values.flatten()], 
                   y=['Efficiency'],
                   x=pivot_eff.index,
                   colorscale='RdYlGn'),
        row=1, col=1
    )
    
    # 2. BPM Box Plot
    for exercise in df['Exercise'].unique():
        fig.add_trace(
            go.Box(y=df[df['Exercise']==exercise]['BPM'], name=exercise),
            row=1, col=2
        )
    
    # 3. Cumulative Calories
    df_sorted = df.sort_values('Calories')
    df_sorted['Cumulative'] = df_sorted['Calories'].cumsum()
    fig.add_trace(
        go.Scatter(x=list(range(len(df_sorted))), y=df_sorted['Cumulative'],
                  mode='lines', fill='tozeroy', line=dict(color='royalblue')),
        row=2, col=1
    )
    
    # 4. Performance Score (composite metric)
    df['Performance'] = (df['Calories']/df['Calories'].max() + 
                        df['Efficiency']/df['Efficiency'].max() + 
                        df['BPM']/df['BPM'].max()) / 3
    perf_avg = df.groupby('Exercise')['Performance'].mean().sort_values()
    fig.add_trace(
        go.Bar(x=perf_avg.values, y=perf_avg.index, orientation='h',
               marker=dict(color=perf_avg.values, colorscale='Plasma')),
        row=2, col=2
    )
    
    fig.update_layout(height=900, showlegend=False,
                     title_text="Advanced Exercise Insights")
    fig.write_html("exercise_advanced_insights.html")
    print("✅ Advanced insights saved: exercise_advanced_insights.html")
    return fig

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("EXERCISE DATA VISUALIZATION - COMPLETE PIPELINE")
    print("="*70)
    
    # 1. Generate/Load Data
    print("\n📊 Step 1: Loading Data...")
    df = pd.read_csv('Copy of finalProj_df - 2022.csv')
    
    # 2. Analyze Data
    print("\n📈 Step 2: Analyzing Data...")
    df = analyze_data(df)
    
    # 3. Create Visualizations
    print("\n🎨 Step 3: Creating Visualizations...")
    
    print("\n[1/5] Creating overview dashboard...")
    create_overview_dashboard(df)
    
    print("\n[2/5] Creating statistical plots...")
    create_statistical_plots(df)
    
    print("\n[3/5] Creating interactive dashboard...")
    create_interactive_dashboard(df)
    
    print("\n[4/5] Creating comparative analysis...")
    create_comparative_analysis(df)
    
    print("\n[5/5] Creating advanced insights...")
    create_advanced_insights(df)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  📁 exercise_dashboard_overview.html")
    print("  📁 exercise_statistical_analysis.png")
    print("  📁 exercise_interactive_scatter.html")
    print("  📁 exercise_comparative_analysis.png")
    print("  📁 exercise_advanced_insights.html")
    print("\n💡 Tip: Open .html files in browser for interactive dashboards")
    print("="*70)

if __name__ == "__main__":
    main()