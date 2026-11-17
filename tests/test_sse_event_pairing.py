import json
import pytest
from llm_shield_proxy.engines.vault import Vault
from llm_shield_proxy.streaming.streaming import rehydrate_sse_stream


def encode(event):
    return ('event:'+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode()


def frames(raw):
    result=[]
    for frame in raw.replace(b'\r\n',b'\n').split(b'\n\n'):
        lines=frame.splitlines()
        names=[line[6:].strip() for line in lines if line.startswith(b'event:')]
        data=[line[5:].strip() for line in lines if line.startswith(b'data:')]
        if names:
            assert data, f'orphan event header: {frame!r}'
            assert len(names)==1, f'mixed event headers: {frame!r}'
            obj=json.loads(b'\n'.join(data))
            assert names[0].decode()==obj['type'], f'event/data mismatch: {frame!r}'
        if data and data[0]!=b'[DONE]': result.append(json.loads(b'\n'.join(data)))
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('width',[1,17,100000])
@pytest.mark.parametrize('signature',[False,True])
async def test_native_thinking_holds_header_and_preserves_event_order(width,signature):
    events=[{'type':'message_start','message':{'id':'fixture','type':'message','role':'assistant','model':'fixture','content':[],'usage':{'input_tokens':1,'output_tokens':0}}},
        {'type':'content_block_start','index':0,'content_block':{'type':'thinking','thinking':'','signature':''}},
        {'type':'content_block_delta','index':0,'delta':{'type':'thinking_delta','thinking':'[EMA'}},
        {'type':'ping'},
        {'type':'content_block_delta','index':0,'delta':{'type':'thinking_delta','thinking':'IL_1]'}}]
    if signature:events.append({'type':'content_block_delta','index':0,'delta':{'type':'signature_delta','signature':'opaque-signature'}})
    events.extend([{'type':'content_block_stop','index':0},
        {'type':'content_block_start','index':1,'content_block':{'type':'text','text':''}},
        {'type':'content_block_delta','index':1,'delta':{'type':'text_delta','text':'星期二'}},
        {'type':'content_block_stop','index':1},
        {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':3}},
        {'type':'message_stop'}])
    raw=b''.join(map(encode,events))
    async def upstream():
        for offset in range(0,len(raw),width):yield raw[offset:offset+width]
    vault=Vault(synthetic=False);vault.get_or_create_token('someone@example.com','EMAIL')
    output=b''.join([piece async for piece in rehydrate_sse_stream(upstream(),vault,anthropic_native=True)])
    parsed=frames(output)
    thinking=[e for e in parsed if (e.get('delta') or {}).get('type')=='thinking_delta']
    assert ''.join(e['delta']['thinking'] for e in thinking)=='someone@example.com'
    assert '[EMAIL_1]' not in output.decode()
    assert parsed[-1]['type']=='message_stop'
    types=[(e['type'],(e.get('delta') or {}).get('type')) for e in parsed]
    if signature:assert types.index(('content_block_delta','thinking_delta'))<types.index(('content_block_delta','signature_delta'))
    assert types.index(('content_block_delta','thinking_delta'))<types.index(('content_block_stop',None))
    assert any((e.get('delta') or {}).get('text')=='星期二' for e in parsed)


@pytest.mark.asyncio
@pytest.mark.parametrize('carrier',['reasoning_content','reasoning'])
async def test_openai_compatible_reasoning_tools_and_done_are_preserved(carrier):
    events=[{'choices':[{'index':0,'delta':{carrier:'[EMAIL_1]'}}]},
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':'send','arguments':'{"to":"[EMA'}}]}}]},
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'IL_1]"}'}}]},'finish_reason':'tool_calls'}]}]
    raw=b''.join(('data:'+json.dumps(e)+'\n\n').encode() for e in events)+b'data: [DONE]\n\n'
    async def upstream():
        for i in range(0,len(raw),7):yield raw[i:i+7]
    vault=Vault(synthetic=False);vault.get_or_create_token('someone@example.com','EMAIL')
    output=b''.join([p async for p in rehydrate_sse_stream(upstream(),vault)])
    parsed=frames(output)
    assert b'event:' not in output
    assert output.endswith(b'\n\n') and b'[DONE]' in output
    assert '[EMAIL_1]' not in output.decode()
    assert ''.join(c.get('delta',{}).get(carrier,'') for e in parsed for c in e.get('choices',[]))=='someone@example.com'
    args=''.join(t['function'].get('arguments','') for e in parsed for c in e.get('choices',[]) for t in c.get('delta',{}).get('tool_calls',[]))
    assert json.loads(args)=={'to':'someone@example.com'}
