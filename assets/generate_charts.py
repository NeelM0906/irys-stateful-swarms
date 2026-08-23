import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica', 'Arial'],
    'font.size': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

IRYS = '#1a73e8'
TENET = '#ea4335'
FABLE = '#9334e6'
OPUS = '#f9ab00'
OTHER = '#aaaaaa'

# ── Chart 1: All-Pass Rate ──
fig, ax = plt.subplots(figsize=(10, 5))
systems = ['irys', 'Muse Spark', 'Tenet', 'Grok 4.5', 'Fable 5', 'Kimi K3', 'DS V4 Flash', 'Opus 4.7', 'Opus 5', 'Gemini 3.6', 'GPT-5.6 Sol']
allpass = [31.6, 20.0, 19.7, 12.9, 11.5, 10.8, 8.3, 7.1, 6.7, 3.3, 2.5]
colors = [IRYS, OTHER, TENET, OTHER, FABLE, OTHER, OTHER, OPUS, OPUS, OTHER, OTHER]

bars = ax.barh(range(len(systems)), allpass, color=colors, height=0.65, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(systems)))
ax.set_yticklabels(systems, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Strict All-Pass Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Harvey Legal Agent Benchmark — All-Pass Rate', fontsize=16, fontweight='bold', pad=15)

for i, (bar, val) in enumerate(zip(bars, allpass)):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', fontsize=11, fontweight='bold' if i == 0 else 'normal',
            color=colors[i])

ax.set_xlim(0, 38)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
plt.tight_layout()
plt.savefig('assets/lab_allpass_rate.png')
plt.close()

# ── Chart 2: Intelligence Per Dollar (bubble/scatter) ──
fig, ax = plt.subplots(figsize=(10, 6))

data = [
    ('irys', 5.11, 31.6, 6.18, IRYS),
    ('Tenet', 8, 19.7, 2.46, TENET),
    ('Fable 5', 102, 11.5, 0.11, FABLE),
    ('Opus 4.7', 51, 7.1, 0.14, OPUS),
]

for name, cost, ap, ipd, color in data:
    size = max(ipd * 120, 80)
    ax.scatter(cost, ap, s=size, c=color, alpha=0.85, edgecolors='white', linewidth=1.5, zorder=5)
    offset_x = 3 if cost < 80 else -8
    offset_y = 1.2
    if name == 'Opus 4.7':
        offset_y = -2.0
    ax.annotate(f'{name}\n{ipd} all-pass/$',
                (cost, ap), textcoords='offset points',
                xytext=(offset_x, offset_y), fontsize=10, fontweight='bold',
                color=color, va='bottom')

ax.set_xlabel('Cost per Task ($)', fontsize=13, fontweight='bold')
ax.set_ylabel('All-Pass Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance vs Cost — Harvey LAB', fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(-5, 120)
ax.set_ylim(0, 40)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

ax.annotate('', xy=(5.11, 31.6), xytext=(102, 11.5),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5, linestyle='--'))
ax.text(55, 24, '56x intelligence\nper dollar', fontsize=10, color='#666666',
        ha='center', style='italic')

plt.tight_layout()
plt.savefig('assets/lab_performance_vs_cost.png')
plt.close()

# ── Chart 3: Cost per All-Pass Point ──
fig, ax = plt.subplots(figsize=(9, 4.5))
systems_cost = ['irys', 'Tenet', 'Opus 4.7', 'Fable 5']
cpp = [0.16, 0.41, 7.18, 8.87]
colors_cost = [IRYS, TENET, OPUS, FABLE]

bars = ax.barh(range(len(systems_cost)), cpp, color=colors_cost, height=0.55, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(systems_cost)))
ax.set_yticklabels(systems_cost, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Cost per All-Pass Percentage Point ($)', fontsize=12, fontweight='bold')
ax.set_title('Cost per Point of Quality', fontsize=16, fontweight='bold', pad=15)

for i, (bar, val) in enumerate(zip(bars, cpp)):
    label = f'${val:.2f}'
    if i == 0:
        label += '  (55x cheaper than Fable 5)'
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=11, fontweight='bold' if i == 0 else 'normal',
            color=colors_cost[i])

ax.set_xlim(0, 14)
plt.tight_layout()
plt.savefig('assets/lab_cost_per_point.png')
plt.close()

# ── Chart 4: What $100 buys ──
fig, ax = plt.subplots(figsize=(9, 4.5))
systems_100 = ['irys', 'Tenet', 'Opus 4.7', 'Fable 5']
expected_pass = [6.2, 2.5, 0.14, 0.11]
colors_100 = [IRYS, TENET, OPUS, FABLE]

bars = ax.barh(range(len(systems_100)), expected_pass, color=colors_100, height=0.55, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(systems_100)))
ax.set_yticklabels(systems_100, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Expected All-Pass Tasks per $100', fontsize=12, fontweight='bold')
ax.set_title('What $100 Buys You on LAB', fontsize=16, fontweight='bold', pad=15)

for i, (bar, val) in enumerate(zip(bars, expected_pass)):
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', fontsize=11, fontweight='bold' if i == 0 else 'normal',
            color=colors_100[i])

ax.set_xlim(0, 8)
plt.tight_layout()
plt.savefig('assets/lab_what_100_buys.png')
plt.close()

# ── Chart 5: Training Investment vs Performance ──
fig, ax = plt.subplots(figsize=(9, 3.5))

ax.barh([0], [31.6], color=IRYS, height=0.5, label='irys-stateful-swarms')
ax.barh([1], [19.7], color=TENET, height=0.5, label='Harvey Tenet')

ax.set_yticks([0, 1])
ax.set_yticklabels([
    'irys\nZero training',
    'Tenet\n150 B300 GPUs × 2 months'
], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('All-Pass Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Training Investment vs Performance', fontsize=16, fontweight='bold', pad=15)

ax.text(31.6 + 0.5, 0, '31.6%', va='center', fontsize=12, fontweight='bold', color=IRYS)
ax.text(19.7 + 0.5, 1, '19.7%', va='center', fontsize=12, fontweight='bold', color=TENET)

ax.annotate('+60%', xy=(25.5, 0.5), fontsize=14, fontweight='bold', color='#333',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe', edgecolor=IRYS, linewidth=1.5))

ax.set_xlim(0, 40)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
plt.tight_layout()
plt.savefig('assets/lab_training_vs_performance.png')
plt.close()

print("All 5 charts generated in assets/")
