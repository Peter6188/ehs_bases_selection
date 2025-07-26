#!/usr/bin/env python3
"""
Summary comparison of all three methods for 100% coverage
"""

import matplotlib.pyplot as plt
import numpy as np

def create_method_comparison_chart():
    """Create a comprehensive comparison chart of all three methods"""
    
    # Method data
    methods = ['Method 1\n(Population-Only)', 'Method 2\n(Hospital Co-located)', 'Method 3\n(Hospital-Performance)']
    bases_95_coverage = [80, 76, 45]  # Bases needed for ~95% coverage
    bases_100_coverage = [80, 76, 126]  # Bases needed for 100% coverage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Bases needed for different coverage levels
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, bases_95_coverage, width, label='~95% Coverage', color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, bases_100_coverage, width, label='100% Coverage', color='darkblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 1, str(int(height1)), 
                ha='center', va='bottom', fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 1, str(int(height2)), 
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of EMS Bases', fontsize=12, fontweight='bold')
    ax1.set_title('EMS Bases Required by Coverage Level', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 140)
    
    # Subplot 2: Efficiency comparison (Coverage per base)
    coverage_per_base_95 = [95/base for base in bases_95_coverage]
    coverage_per_base_100 = [100/base for base in bases_100_coverage]
    
    bars3 = ax2.bar(x - width/2, coverage_per_base_95, width, label='At ~95% Coverage', color='lightgreen', alpha=0.8)
    bars4 = ax2.bar(x + width/2, coverage_per_base_100, width, label='At 100% Coverage', color='darkgreen', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        ax2.text(bar3.get_x() + bar3.get_width()/2., height3 + 0.02, f'{height3:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(bar4.get_x() + bar4.get_width()/2., height4 + 0.02, f'{height4:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coverage % per Base', fontsize=12, fontweight='bold')
    ax2.set_title('Efficiency: Coverage Percentage per Base', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig('../../data/processed/method_comparison_100percent_coverage.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison chart saved to: ../../data/processed/method_comparison_100percent_coverage.png")
    plt.show()

def print_summary():
    """Print a comprehensive summary of findings"""
    print("🎯 METHOD COMPARISON: 100% COVERAGE ANALYSIS")
    print("=" * 60)
    print()
    print("📊 BASES REQUIRED FOR 100% COVERAGE:")
    print("   Method 1 (Population-Only):      80 bases")
    print("   Method 2 (Hospital Co-located):  76 bases") 
    print("   Method 3 (Hospital-Performance): 126 bases")
    print()
    print("📊 KEY INSIGHTS:")
    print("   • Method 2 is MOST EFFICIENT for 100% coverage (76 bases)")
    print("   • Method 3 requires 66% MORE bases than Method 2 (126 vs 76)")
    print("   • Method 3 requires 58% MORE bases than Method 1 (126 vs 80)")
    print()
    print("🎯 STRATEGIC RECOMMENDATIONS:")
    print("   • For 100% coverage: Use Method 2 (Hospital Co-located)")
    print("   • For 95% coverage: Use Method 3 (Hospital-Performance)")
    print("   • Method 3 trades coverage efficiency for performance optimization")
    print()
    print("📈 EFFICIENCY ANALYSIS:")
    print("   • Method 2: 1.32% coverage per base")
    print("   • Method 1: 1.25% coverage per base") 
    print("   • Method 3: 0.79% coverage per base")
    print()
    print("🔍 METHOD 3 ADDITIONAL BASES:")
    print("   • Current Method 3: 45 bases (96.7% coverage)")
    print("   • Additional needed: 81 bases") 
    print("   • Total for 100%: 126 bases")

if __name__ == "__main__":
    print_summary()
    create_method_comparison_chart()
