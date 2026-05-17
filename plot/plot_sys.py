import graphviz
dot = graphviz.Digraph(comment='System Architecture', format='pdf')
dot.attr(rankdir='TB', size='10,10', dpi='300') # TB: Top to Bottom

# 设置全局节点样式 (学术风格：清晰的边框、柔和的填充色)
dot.attr('node', shape='box', style='rounded,filled', fillcolor='aliceblue', 
         fontname='Helvetica', fontsize='12', penwidth='1.2')
dot.attr('edge', fontname='Helvetica', fontsize='10', penwidth='1.2')

# ==================== Input ====================
dot.node('Input', 'Input: Climate-related Claim', 
         shape='parallelogram', fillcolor='lightyellow', fontname='Helvetica-Bold')

# ==================== Phase 1: RAG Pipeline ====================
with dot.subgraph(name='cluster_phase1') as c:
    c.attr(label='Phase 1: Two-Stage RAG Pipeline', style='dashed', color='gray', fontname='Helvetica-Bold', fontsize='14')
    
    # 定义节点
    c.node('KB', 'Knowledge Corpus\n(evidence.json)', shape='cylinder', fillcolor='lightgrey')
    c.node('FAISS', 'FAISS Vector Database')
    c.node('BiEnc', 'Initial Recall: Bi-Encoder\n(all-MiniLM-L6-v2)')
    c.node('Top20', 'Top-20 Candidate Passages')
    c.node('CrossEnc', 'Precision Re-ranking: Cross-Encoder\n(ms-marco-MiniLM-L-6-v2)')
    c.node('Filter', 'Threshold Filtering\n(Score ≥ 2.5, Max 5 Passages)')
    c.node('Context', 'Filtered Evidence Context', shape='note', fillcolor='lightcyan')

    # 定义 Phase 1 内部连接
    c.edge('KB', 'FAISS', style='dashed', label=' Indexing')
    c.edge('FAISS', 'BiEnc', label=' Similarity Search')
    c.edge('BiEnc', 'Top20')
    c.edge('Top20', 'CrossEnc')
    c.edge('CrossEnc', 'Filter')
    c.edge('Filter', 'Context')

# ==================== Phase 2: LLM Reasoning ====================
with dot.subgraph(name='cluster_phase2') as c:
    c.attr(label='Phase 2: Reasoning & Classification', style='dashed', color='gray', fontname='Helvetica-Bold', fontsize='14')
    
    # 定义节点
    c.node('Prompt', 'Prompt Formulation\n(Concatenate Claim & Context)')
    c.node('LLM', 'LoRA-adapted LLM\n(Qwen3.5-2B, 4-bit)')
    c.node('CoT', 'Chain-of-Thought (CoT) Reasoning', shape='note', fillcolor='lightyellow')
    
    # 定义 Phase 2 内部连接
    c.edge('Prompt', 'LLM')
    c.edge('LLM', 'CoT')

# ==================== Output ====================
dot.node('Regex', 'Label Extraction\n(Regular Expression)')
dot.node('Label', 'Final Label\n{SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}', 
         shape='parallelogram', fillcolor='palegreen', fontname='Helvetica-Bold')

# ==================== Cross-Cluster Edges ====================
# 连接输入到管线
dot.edge('Input', 'BiEnc')
dot.edge('Input', 'Prompt')

# 两个阶段的衔接
dot.edge('Context', 'Prompt')

# 最终输出逻辑
dot.edge('CoT', 'Regex')
dot.edge('Regex', 'Label')

# ==================== 渲染与保存 ====================
# 运行后会在当前目录生成 system_architecture.pdf
dot.render('system_architecture', view=True)
print("Flowchart has been saved as 'system_architecture.pdf'")