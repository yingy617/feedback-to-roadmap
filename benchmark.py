"""Measure latency, tokens and cost of the pipeline, sequential vs parallel.

Usage: python benchmark.py [feedback_file] [parallel_workers]
Makes real Claude API calls (roughly 1 + 4 x themes calls; a few cents).
"""
import sys
import json
import time
import app as A

path = sys.argv[1] if len(sys.argv) > 1 else 'sample_feedback.txt'
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
items = A.parse_feedback(open(path, encoding='utf-8').read())

m = A.RunMetrics('cluster')
themes = A.cluster_and_tag_feedback(items, m)['themes']
cluster = m.summary()
print(f'{len(items)} items -> {len(themes)} themes, cluster {cluster["wall_ms"]/1000:.1f}s')


def run(w):
    t = json.loads(json.dumps(themes))
    t0 = time.perf_counter()
    rec = A.RunMetrics('recommend')
    A.recommend_all(t, items, rec, workers=w)
    r = rec.summary()
    gr = A.RunMetrics('grounding')  # created after recommend so its timer covers only grounding
    A.ground_all(t, items, gr, workers=w)
    g = gr.summary()
    total = time.perf_counter() - t0
    qc = [c for th in t for c in th.get('quote_checks', [])]
    return {
        'workers': w, 'recommend_s': r['wall_ms'] / 1000, 'grounding_s': g['wall_ms'] / 1000,
        'total_s': round(total, 2), 'calls': r['llm_calls'] + g['llm_calls'],
        'tokens_in': r['input_tokens'] + g['input_tokens'], 'tokens_out': r['output_tokens'] + g['output_tokens'],
        'cost_usd': round(r['cost_usd'] + g['cost_usd'], 4),
        'quotes': len(qc), 'quotes_not_in_source': sum(1 for c in qc if not c['verified']),
        'llm_grounding_skipped': sum(1 for th in t if not th.get('llm_checked')),
    }


results = [run(1), run(workers)]
for r in results:
    print(json.dumps(r))
seq, par = results
print(f"\nRecommend + grounding for {len(themes)} themes: {seq['total_s']:.1f}s sequential -> "
      f"{par['total_s']:.1f}s with {workers} workers ({seq['total_s']/par['total_s']:.1f}x faster)")
full_cost = cluster['cost_usd'] + par['cost_usd']
full_time = cluster['wall_ms'] / 1000 + par['total_s']
print(f"Full run (cluster + recommend + grounding): {full_time:.1f}s, "
      f"{cluster['input_tokens'] + par['tokens_in']:,} in / {cluster['output_tokens'] + par['tokens_out']:,} out tokens, "
      f"~${full_cost:.4f}")
