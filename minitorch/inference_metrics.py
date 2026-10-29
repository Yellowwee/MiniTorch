"""Comparable request-local KV and shared-pool byte accounting (no GPU reads)."""
KV_FIELDS=('kv_live_bytes','kv_capacity_bytes','kv_unused_bytes','kv_metadata_bytes',
    'kv_allocator_slack_bytes','kv_pool_reserved_bytes','kv_pool_metadata_bytes',
    'kv_pool_assigned_bytes','kv_pool_free_bytes','kv_pool_used_pages','kv_pool_free_pages',
    'kv_pool_max_pages','kv_pool_page_acquires','kv_pool_page_releases','kv_pool_page_reuses')


def cache_memory(cache):
    result=dict.fromkeys(KV_FIELDS,0)
    result['kv_slot_utilization']=None
    if cache is not None:
        result.update({k:v for k,v in cache.memory_stats().items() if k in result})
    result['kv_request_allocated_bytes']=(result['kv_capacity_bytes']+
        result['kv_allocator_slack_bytes']+result['kv_metadata_bytes'])
    result['kv_total_reserved_bytes']=(result['kv_pool_reserved_bytes']+
        result['kv_pool_metadata_bytes']+result['kv_metadata_bytes'] if hasattr(cache,'pool')
        else result['kv_request_allocated_bytes'])
    return result


def shared_cache_memory(caches,pool=None):
    """Sum exclusive cache state; count a shared physical pool exactly once."""
    rows=[cache_memory(c) for c in caches if c is not None]
    result=dict.fromkeys(KV_FIELDS,0)
    for key in ('kv_live_bytes','kv_capacity_bytes','kv_unused_bytes','kv_metadata_bytes','kv_allocator_slack_bytes'):
        result[key]=sum(r[key] for r in rows)
    if pool is not None:
        result.update({k:v for k,v in pool.memory_stats().items() if k in result})
    cap=result['kv_capacity_bytes']
    result['kv_slot_utilization']=result['kv_live_bytes']/cap if cap else None
    result['kv_total_reserved_bytes']=(result['kv_pool_reserved_bytes']+result['kv_pool_metadata_bytes']+
        result['kv_metadata_bytes'] if pool is not None else cap+result['kv_allocator_slack_bytes']+result['kv_metadata_bytes'])
    return result
