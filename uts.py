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
# sns.set_palette("viridis")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)

# ==================== DATA ANALYSIS & PREP ====================
def analyze_data(df):
    """Statistical analysis & data preparation for Transaction Data"""
    
    # 1. Data Cleaning & Preparation
    # Mengisi NaN pada kolom 'category' atau 'sku_name' jika ada
    df['category'] = df['category'].fillna('Unknown')
    df['sku_name'] = df['sku_name'].fillna('Unknown')

    # Membuat metrik baru 'avg_price' (pengganti Efficiency)
    # Gunakan harga setelah diskon dibagi kuantitas untuk mendapatkan harga rata-rata per unit/transaksi.
    # Tambahkan penanganan error ZeroDivisionError
    df['avg_price'] = np.where(df['qty_ordered'] > 0, 
                                df['after_discount'] / df['qty_ordered'], 
                                df['price']) # Jika qty 0, pakai harga satuan
    
    # Konversi kolom tanggal jika ada
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        # Membuat kolom bulanan (untuk visualisasi trend, pengganti TED Talks Time Series)
        df['Order_Month'] = df['order_date'].dt.to_period('M')

    print("="*70)
    print("TRANSACTION DATA ANALYSIS REPORT")
    print("="*70)
    
    print("\n1. DATASET OVERVIEW")
    print(f"Total Transactions: {len(df)}")
    print(f"Features: {df.shape[1]}")
    # Menggunakan 'category' sebagai pengganti 'Exercise'
    print(f"\nTotal Categories: {df['category'].nunique()}") 
    print(df['category'].value_counts().head(5))
    
    print("\n2. DESCRIPTIVE STATISTICS")
    # Menggunakan kolom numerik yang relevan
    print(df[['price', 'qty_ordered', 'after_discount', 'avg_price', 'discount_amount']].describe())
    
    print("\n3. CORRELATION ANALYSIS (HR Analytics Style)")
    numeric_cols = ['price', 'qty_ordered', 'after_discount', 'avg_price', 'discount_amount']
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    print("\n4. KEY INSIGHTS")
    # Insight berdasarkan data transaksi
    print(f"Highest Avg Revenue: {df.groupby('category')['after_discount'].mean().idxmax()}")
    print(f"Highest Avg Order Qty: {df.groupby('category')['qty_ordered'].mean().idxmax()}")
    print(f"Highest Avg Price: {df.groupby('category')['avg_price'].mean().idxmax()}")
    
    return df

# ==================== VISUALIZATION 1: Overview Dashboard (Netflix Style) ====================
def create_overview_dashboard(df):
    """Create comprehensive overview - using category, after_discount, price, qty_ordered"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Category Distribution (Volume)', 'Average Revenue by Category',
                        'Price vs Revenue', 'Quantity Ordered Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'box'}]]
    )
    
    # 1. Pie Chart - Category Distribution (by Volume/Qty)
    category_qty = df.groupby('category')['qty_ordered'].sum().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Pie(labels=category_qty.index, values=category_qty.values,
               hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)),
        row=1, col=1
    )
    
    # 2. Bar Chart - Average Revenue (after_discount)
    avg_rev = df.groupby('category')['after_discount'].mean().sort_values(ascending=True).tail(10)
    fig.add_trace(
        go.Bar(x=avg_rev.values, y=avg_rev.index, orientation='h',
               marker=dict(color=avg_rev.values, colorscale='Oranges')),
        row=1, col=2
    )
    
    # 3. Scatter - Price vs Revenue (after_discount)
    fig.add_trace(
        go.Scatter(x=df['price'], y=df['after_discount'], mode='markers',
                     marker=dict(size=8, color=df['qty_ordered'], colorscale='Viridis',
                                 showscale=True, colorbar=dict(title="Qty Ordered")),
                     text=df['category']),
        row=2, col=1
    )
    
    # 4. Box Plot - Quantity Ordered Distribution
    for cat in df['category'].unique()[:10]: # Batasi 10 kategori teratas
        cat_data = df[df['category'] == cat]['qty_ordered']
        fig.add_trace(
            go.Box(y=cat_data, name=cat, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, 
                      title_text="Transaction Analytics Dashboard - Overview",
                      title_font_size=20)
    fig.write_html("transaction_dashboard_overview.html")
    print("\n✅ Overview dashboard saved: transaction_dashboard_overview.html")
    return fig

# ==================== VISUALIZATION 2: Statistical Analysis (Penguins Style) ====================
def create_statistical_plots(df):
    """Deep statistical analysis - from Penguins notebook style"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Analysis of Transaction Data', fontsize=18, fontweight='bold')
    
    # 1. Correlation Heatmap
    numeric_cols = ['price', 'qty_ordered', 'after_discount', 'avg_price']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                 square=True, ax=axes[0,0], cbar_kws={'label': 'Correlation'})
    axes[0,0].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 2. Distribution with KDE
    for col in ['after_discount', 'price']:
        sns.histplot(data=df, x=col, kde=True, ax=axes[0,1], alpha=0.5, label=col)
    axes[0,1].set_title('Revenue & Price Distribution', fontsize=14, fontweight='bold')
    axes[0,1].legend()
    
    # 3. Violin Plot - Quantity Ordered by Category
    sns.violinplot(data=df.head(500), x='category', y='qty_ordered', ax=axes[1,0], palette='muted')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha='right')
    axes[1,0].set_title('Quantity Ordered Distribution by Category', fontsize=14, fontweight='bold')
    
    # 4. Average Price Comparison (Efficiency replacement)
    avg_price_cat = df.groupby('category')['avg_price'].mean().sort_values().tail(10)
    avg_price_cat.plot(kind='barh', ax=axes[1,1], color='teal', alpha=0.7)
    axes[1,1].set_title('Average Transaction Price by Category', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Average Price')
    
    plt.tight_layout()
    plt.savefig('transaction_statistical_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Statistical plots saved: transaction_statistical_analysis.png")
    plt.close(fig) # Menggunakan plt.close(fig) alih-alih plt.show() untuk lingkungan non-interaktif

# ==================== VISUALIZATION 3: Interactive Plotly Dashboard (YouTube Music Style) ====================
def create_interactive_dashboard(df):
    """Interactive dashboard - from YouTube Music notebook style"""
    fig = go.Figure()
    
    # Create traces for each top 10 category
    top_categories = df['category'].value_counts().head(10).index
    for cat in top_categories:
        cat_df = df[df['category'] == cat]
        fig.add_trace(go.Scatter(
            x=cat_df['qty_ordered'],
            y=cat_df['after_discount'],
            mode='markers',
            name=cat,
            marker=dict(size=cat_df['price']/1000, opacity=0.6, sizemode='area', sizeref=2.*max(cat_df['price']/1000)/(40**2)), # marker size based on price
            text=[f"{cat}<br>Price: {p}<br>Revenue: {rev}" 
                  for p, rev in zip(cat_df['price'], cat_df['after_discount'])],
            hovertemplate='<b>%{text}</b><br>Qty: %{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Interactive Transaction Analysis: Quantity vs Revenue (Size = Price)',
        xaxis_title='Quantity Ordered',
        yaxis_title='Revenue (After Discount)',
        hovermode='closest',
        height=600,
        template='plotly_white'
    )
    
    fig.write_html("transaction_interactive_scatter.html")
    print("✅ Interactive scatter saved: transaction_interactive_scatter.html")
    return fig

# ==================== VISUALIZATION 4: Comparative Analysis (NBA Style) ====================
def create_comparative_analysis(df):
    """Multi-metric comparison - from NBA notebook style"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparative Transaction Analysis', fontsize=18, fontweight='bold')
    
    # 1. Grouped Bar - Multiple Metrics (Top 10 Category)
    metrics_df = df.groupby('category').agg({
        'after_discount': 'mean',
        'qty_ordered': 'mean',
        'price': 'mean'
    }).loc[df['category'].value_counts().head(10).index].reset_index()
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    axes[0,0].bar(x - width, metrics_df['after_discount']/10000, width, label='Revenue/10k', alpha=0.8)
    axes[0,0].bar(x, metrics_df['qty_ordered']*10, width, label='Qty Ordered x10', alpha=0.8) # Kali 10 untuk normalisasi visual
    axes[0,0].bar(x + width, metrics_df['price']/1000, width, label='Price/1k', alpha=0.8)
    axes[0,0].set_xlabel('Category')
    axes[0,0].set_ylabel('Value (Normalized)')
    axes[0,0].set_title('Multi-Metric Comparison (Top 10)')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(metrics_df['category'], rotation=45, ha='right')
    axes[0,0].legend()
    
    # 2. Box Plot - Revenue Distribution
    df.boxplot(column='after_discount', by='category', ax=axes[0,1])
    axes[0,1].set_title('Revenue Distribution by Category')
    axes[0,1].set_xlabel('Category')
    axes[0,1].set_ylabel('Revenue')
    plt.sca(axes[0,1])
    plt.xticks(rotation=45, ha='right')
    
    # 3. Avg Price Ranking (Efficiency replacement)
    avg_price_stats = df.groupby('category')['avg_price'].agg(['mean', 'std']).sort_values('mean').tail(10)
    avg_price_stats['mean'].plot(kind='barh', xerr=avg_price_stats['std'], ax=axes[1,0], 
                             color='coral', alpha=0.7, error_kw={'elinewidth': 2})
    axes[1,0].set_title('Category Price with Standard Deviation (Top 10)')
    axes[1,0].set_xlabel('Average Price')
    
    # 4. Count Plot (Top 10)
    category_counts = df['category'].value_counts().head(10)
    axes[1,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                  startangle=90, colors=sns.color_palette('pastel'))
    axes[1,1].set_title('Category Type Distribution (Top 10)')
    
    plt.tight_layout()
    plt.savefig('transaction_comparative_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Comparative analysis saved: transaction_comparative_analysis.png")
    plt.close(fig)

# ==================== VISUALIZATION 5: Advanced Insights (TED Talks & Crypto Style) ====================
def create_advanced_insights(df):
    """Advanced analytics - focusing on Time Series & Performance Score"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Price Heatmap', 'Revenue Range by Category',
                        'Revenue Trend Over Time (Time Series)', 'Category Performance Score'),
        specs=[[{'type': 'heatmap'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Average Price Heatmap (Efficiency replacement)
    # Gunakan pivot table sederhana untuk harga rata-rata per kategori
    pivot_price = df.groupby('category')['avg_price'].mean().sort_values()
    fig.add_trace(
        go.Heatmap(z=[pivot_price.values.flatten()], 
                    y=['Avg Price'],
                    x=pivot_price.index,
                    colorscale='RdYlGn'),
        row=1, col=1
    )
    
    # 2. Revenue (after_discount) Box Plot
    for cat in df['category'].unique()[:10]:
        fig.add_trace(
            go.Box(y=df[df['category']==cat]['after_discount'], name=cat),
            row=1, col=2
        )
    
    # 3. Revenue Cumulative Trend (Time Series - TED Talks Style)
    if 'Order_Month' in df.columns:
        monthly_revenue = df.groupby('Order_Month')['after_discount'].sum().cumsum().reset_index()
        monthly_revenue['Order_Month'] = monthly_revenue['Order_Month'].astype(str)
        fig.add_trace(
            go.Scatter(x=monthly_revenue['Order_Month'], y=monthly_revenue['after_discount'],
                       mode='lines+markers', line=dict(color='royalblue')),
            row=2, col=1
        )
    else:
        fig.update_annotations(text='No Time Series Data', row=2, col=1)

    
    # 4. Performance Score (composite metric - Crypto Style)
    # Normalisasi metrik: Revenue, Kuantitas, dan Harga
    df['Performance'] = (df['after_discount']/df['after_discount'].max() + 
                         df['qty_ordered']/df['qty_ordered'].max() + 
                         df['price']/df['price'].max()) / 3
    perf_avg = df.groupby('category')['Performance'].mean().sort_values().tail(10)
    fig.add_trace(
        go.Bar(x=perf_avg.values, y=perf_avg.index, orientation='h',
               marker=dict(color=perf_avg.values, colorscale='Plasma')),
        row=2, col=2
    )
    
    fig.update_layout(height=900, showlegend=False,
                      title_text="Advanced Transaction Insights")
    fig.write_html("transaction_advanced_insights.html")
    print("✅ Advanced insights saved: transaction_advanced_insights.html")
    return fig

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("TRANSACTION DATA VISUALIZATION - COMPLETE PIPELINE")
    print("="*70)
    
    # 1. Load Data
    print("\n📊 Step 1: Loading Data...")
    # PASTIKAN FILE 'Copy of finalProj_df - 2022.csv' ADA di direktori yang sama
    try:
        df = pd.read_csv('Copy of finalProj_df - 2022.csv')
        # Hapus kolom 'registered_date,' yang kosong atau tidak relevan
        if 'registered_date,' in df.columns:
            df = df.drop(columns=['registered_date,'])
    except FileNotFoundError:
        print("❌ ERROR: File 'Copy of finalProj_df - 2022.csv' not found.")
        print("Please ensure the file is in the same directory as the script.")
        return # Hentikan eksekusi jika file tidak ditemukan
    
    # 2. Analyze Data
    print("\n📈 Step 2: Analyzing Data...")
    df = analyze_data(df)
    
    # 3. Create Visualizations
    print("\n🎨 Step 3: Creating Visualizations...")
    
    print("\n[1/5] Creating overview dashboard (Netflix Style)...")
    create_overview_dashboard(df)
    
    print("\n[2/5] Creating statistical plots (Penguins Style)...")
    create_statistical_plots(df)
    
    print("\n[3/5] Creating interactive dashboard (YouTube Music Style)...")
    create_interactive_dashboard(df)
    
    print("\n[4/5] Creating comparative analysis (NBA Style)...")
    create_comparative_analysis(df)
    
    print("\n[5/5] Creating advanced insights (TED Talks & Crypto Style)...")
    create_advanced_insights(df)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  📁 transaction_dashboard_overview.html")
    print("  📁 transaction_statistical_analysis.png")
    print("  📁 transaction_interactive_scatter.html")
    print("  📁 transaction_comparative_analysis.png")
    print("  📁 transaction_advanced_insights.html")
    print("\n💡 Tip: Open the .html files in your browser for the interactive dashboards.")
    print("="*70)

if __name__ == "__main__":
    main()
