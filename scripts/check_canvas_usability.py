"""Check layer navigation and contextual insertion without saving or training."""
from playwright.sync_api import sync_playwright, expect
from check_canvas_insertion import EditorCheck, args, edge_id


def check_usability(page, url):
    check = EditorCheck(page)
    page.goto(url)
    original = check.graph()
    page.wait_for_timeout(800)
    assert page.get_by_label('Search layers', exact=True).bounding_box()['y'] < page.get_by_label('Graph name', exact=True).bounding_box()['y']
    expect(page.locator('.node-insert')).not_to_have_count(0)
    page.get_by_role('button', name='Hide left panel', exact=True).click()
    page.keyboard.press('Control+k')
    expect(page.get_by_role('complementary', name='Workspace controls')).to_be_visible()
    expect(page.get_by_label('Search layers', exact=True)).to_be_focused()
    page.get_by_label('Search layers', exact=True).fill('no-such-layer')
    expect(page.get_by_text('No matching layers. Try a name such as Dense or LSTM.')).to_be_visible()
    page.get_by_role('button', name='Clear layer search', exact=True).click()

    page.locator('[data-block-id="group:layer-core"] .architecture-node-title strong').click()
    count = page.locator('.stage-layer-list .architecture-tree-row').count()
    assert 0 < count < len([n for n in original['nodes'] if n.get('group') == 'layer-core'])
    assert page.locator('.stage-tensor-operations').get_attribute('open') is None
    page.get_by_role('button', name='Inspect main layer', exact=True).click()
    picker = page.get_by_label('Layer in stage', exact=True)
    first = picker.input_value()
    page.get_by_role('button', name='Next layer', exact=True).click()
    expect(picker).not_to_have_value(first)
    page.get_by_role('button', name='Previous layer', exact=True).click()
    expect(picker).to_have_value(first)
    page.wait_for_timeout(500)
    sidebar = page.get_by_role('complementary', name='Workspace controls')
    sidebar.evaluate('el => { el.scrollTop = el.scrollHeight }')
    assert sidebar.evaluate('el => el.scrollTop') > 0
    page.locator('.architecture-node.is-selected .architecture-node-title strong').click()
    assert sidebar.evaluate('el => el.scrollTop') == 0
    page.get_by_role('button', name='Fit diagram', exact=True).click()
    page.get_by_role('button', name='Locate on canvas', exact=True).click()
    assert float(page.locator('.architecture-diagram').get_attribute('data-zoom')) > 1

    edge = next(e for e in original['edges'] if e['source'] == first)
    plus = page.locator('.node-insert').filter(has=page.locator('svg'))
    plus = next(plus.nth(i) for i in range(plus.count()) if plus.nth(i).get_attribute('data-edge-id') == edge_id(edge))
    plus.click()
    expect(page.get_by_label('Insertion destination')).to_have_value(edge_id(edge))
    page.get_by_label('Search layer palette').fill('Dropout')
    page.get_by_title('Add Dropout', exact=True).click()
    inserted = check.graph(lambda g: len(g['nodes']) == len(original['nodes']) + 1)
    new_id = next(n['id'] for n in inserted['nodes'] if n['id'] not in {n['id'] for n in original['nodes']})
    assert {'source': first, 'target': new_id, 'port': 'x'} in inserted['edges']
    assert {**edge, 'source': new_id} in inserted['edges']
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['edges'] == original['edges'])

    # Pick one branch explicitly and leave all other outgoing connections alone.
    source = next(n['id'] for n in original['nodes'] if sum(e['source'] == n['id'] for e in original['edges']) > 1)
    check.find(source)
    choices = [e for e in original['edges'] if e['source'] == source]
    check.tab('Add')
    page.get_by_label('Insertion destination').select_option(edge_id(choices[-1]))
    page.get_by_label('Search layer palette').fill('Dropout')
    page.get_by_title('Add Dropout', exact=True).click()
    branched = check.graph(lambda g: len(g['nodes']) == len(original['nodes']) + 1)
    for unaffected in choices[:-1]:
        assert unaffected in branched['edges']
    assert choices[-1] not in branched['edges']
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['edges'] == original['edges'])

    # Changing the type of an existing executable layer retains its slot and every branch.
    check.find(first)
    page.get_by_label('Layer type', exact=True).select_option('dropout')
    replaced = check.graph(lambda g: not any(n['id'] == first for n in g['nodes']))
    position = next(i for i, n in enumerate(original['nodes']) if n['id'] == first)
    replacement = replaced['nodes'][position]
    assert replacement['kind'] == 'dropout'
    assert replacement.get('group') == original['nodes'][position].get('group')
    assert len(replaced['nodes']) == len(original['nodes'])
    assert replaced['edges'] == [
        {**e, 'source': replacement['id'] if e['source'] == first else e['source'],
         'target': replacement['id'] if e['target'] == first else e['target'],
         'port': 'x' if e['target'] == first else e['port']}
        for e in original['edges']
    ]
    expect(page.get_by_label('Layer type', exact=True)).to_have_value('dropout')
    page.screenshot(path='/tmp/hypercast-selected-layer-type.png')
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['nodes'] == original['nodes'] and g['edges'] == original['edges'])

    page.get_by_role('button', name='Fit diagram', exact=True).click()
    page.wait_for_timeout(600)
    viewport = page.locator('.react-flow__viewport')
    before = viewport.get_attribute('style')
    mini = page.locator('.architecture-minimap')
    mini.click(position={'x': 25, 'y': 15})
    expect(viewport).not_to_have_attribute('style', before)
    page.get_by_role('button', name='Reset layout', exact=True).click()
    page.wait_for_timeout(500)
    page.screenshot(path='/tmp/hypercast-easier-editor.png')
    assert check.graph()['edges'] == original['edges']
    assert not check.errors, check.errors
    assert not check.jobs, check.jobs
    print('PASS: visible insertion buttons, early search and shortcut, concise stages, previous/next layers, locate, automatic insertion, explicit branch choice, layer type replacement with branches preserved, minimap navigation and unchanged graph after undo.')


if __name__ == '__main__':
    options = args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': options.browser_binary} if options.browser_binary else {}))
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        check_usability(page, options.url)
        browser.close()
