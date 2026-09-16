"""Real-browser node editor acceptance checks. No training or persisted writes.

Requires Playwright and Chromium.
Usage: python scripts/check_canvas.py [url] [--browser-binary PATH]
"""
import json
from playwright.sync_api import expect, sync_playwright
from check_canvas_insertion import EditorCheck, args, edge_id
def geometry(page):
    return page.locator('.architecture-node').evaluate_all("els => els.map(el => ({id: el.dataset.blockId}))")


def click_node(page, block):
    page.locator(f'[data-block-id="{block["id"]}"] .architecture-node-title strong').click()


def check_overview(page, url):
    check = EditorCheck(page)
    page.goto(url)
    check.graph()
    page.get_by_label('Load graph preset').select_option('paper-cnn')
    original = check.graph(lambda g: g.get('preset_id') == 'paper-cnn')
    page.get_by_role('button', name='Clone to edit', exact=True).click()
    check.graph(lambda g: not g.get('locked'))
    combined = {n['id'] for n in original['nodes'] if n.get('group') in ['layer-first', 'layer-first-relu']}
    page.wait_for_timeout(500)
    click_node(page, next(b for b in geometry(page) if b['id'] == 'group:layer-first'))
    expect(page.get_by_role('tab', name='Inspect', exact=True)).to_have_attribute('aria-selected', 'true')
    page.get_by_role('button', name='Duplicate independently', exact=True).click()
    duplicate = check.graph(lambda g: len(g['nodes']) == len(original['nodes']) + len(combined))
    assert {n['source_ref']['node'] for n in duplicate['nodes'] if n['id'] not in {n['id'] for n in original['nodes']}} == combined
    expect(page.locator('.disconnected-label')).to_be_visible()
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['nodes'] == original['nodes'])
    page.get_by_role('button', name='Delete selected', exact=True).click()
    deleted = check.graph(lambda g: len(g['nodes']) == len(original['nodes']) - len(combined))
    assert {n['id'] for n in deleted['nodes']} == {n['id'] for n in original['nodes']} - combined
    assert deleted['edges'] == [e for e in original['edges'] if e['source'] not in combined and e['target'] not in combined]
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['nodes'] == original['nodes'])
    # Search crosses summary boundaries and exposes the exact operation.
    activation = next(n['id'] for n in original['nodes'] if n.get('group') == 'layer-first-relu')
    check.find(activation)
    expect(page.locator(f'[data-block-id="{activation}"]')).to_have_attribute('aria-pressed', 'true')
    check.tab('Architecture')
    page.get_by_role('button', name='Save graph', exact=True).click()
    expect(page.get_by_text('Graph and canvas layout saved.', exact=True)).to_be_visible()
    saved = check.saved[-1]
    assert saved['nodes'] == original['nodes'] and saved['edges'] == original['edges'] and saved['groups'] == original['groups']
    assert 'layer-first' in saved['view']['expandedStages']
    zoom = page.locator('.architecture-diagram').get_attribute('data-zoom')
    validation_count = len(check.submitted)
    page.get_by_label('Import architecture').set_input_files({'name': 'overview.json', 'mimeType': 'application/json', 'buffer': json.dumps(saved).encode()})
    check.graph(lambda _: len(check.submitted) > validation_count)
    expect(page.locator('.diagram-stage-breadcrumbs').get_by_role('button', name='Collapse Conv1D + ReLU', exact=True)).to_be_visible()
    expect(page.locator('.architecture-diagram')).to_have_attribute('data-zoom', zoom)
    assert not check.errors, check.errors
    assert not check.jobs, check.jobs
    print('PASS: summary selection, exact combined-stage duplication/deletion, tree expansion, original graph saved unchanged, summary/camera import round trip.')


def check_view(page, url):
    check = EditorCheck(page)
    page.goto(url)
    original = check.graph()
    diagram = page.locator('.architecture-diagram')
    expect(page.locator('.architecture-diagram .react-flow')).to_be_visible()
    page.wait_for_timeout(500)
    initial_zoom = float(diagram.get_attribute('data-zoom'))
    page.get_by_role('button', name='Zoom in canvas', exact=True).click()
    expect(diagram).not_to_have_attribute('data-zoom', f'{initial_zoom:.4f}')
    page.get_by_role('button', name='Zoom out canvas', exact=True).click()
    assert abs(float(diagram.get_attribute('data-zoom')) - initial_zoom) < .0002
    page.get_by_role('button', name='Hide left panel', exact=True).click()
    page.wait_for_timeout(250)
    # Select a visible node and automatically reveal the sidebar.
    first = page.locator('.architecture-node').first
    click_node(page, geometry(page)[0])
    expect(page.get_by_role('tab', name='Inspect', exact=True)).to_have_attribute('aria-selected', 'true')
    expect(page.get_by_role('complementary', name='Workspace controls')).to_be_visible()
    expect(first).to_have_attribute('aria-pressed', 'true')
    page.wait_for_timeout(350)

    box = diagram.bounding_box()
    before = first.bounding_box()
    page.mouse.move(box['x'] + box['width'] - 100, box['y'] + 130)
    page.mouse.down()
    page.mouse.move(box['x'] + box['width'] - 170, box['y'] + 165, steps=8)
    page.mouse.up()
    page.wait_for_timeout(150)
    after = first.bounding_box()
    assert abs(after['x'] - before['x'] + 70) < 2, (before, after)
    zoom = diagram.get_attribute('data-zoom')
    for name in ['Runs', 'Compare']:
        page.get_by_role('button', name=name, exact=True).click()
        expect(diagram).not_to_be_visible()
        page.get_by_role('button', name='Builder', exact=True).click()
        expect(diagram).to_have_attribute('data-zoom', zoom)
        expect(first).to_have_attribute('aria-pressed', 'true')
        assert abs(first.bounding_box()['x'] - after['x']) < 2

    # Resizing preserves the camera and is keyboard accessible.
    resizer = page.get_by_role('separator', name='Resize left panel')
    resizer.focus()
    resizer.press('ArrowRight')
    expect(resizer).to_have_attribute('aria-valuenow', '336')
    resizer.press('Home')
    expect(resizer).to_have_attribute('aria-valuenow', '320')
    page.get_by_role('button', name='Fit diagram', exact=True).click()
    page.locator('.node-expand').first.click()
    expect(page.locator('.diagram-stage-breadcrumbs button').first).to_be_visible()
    page.get_by_role('button', name='Reset layout', exact=True).click()
    expect(page.locator('.diagram-stage-breadcrumbs button')).to_have_count(0)
    assert check.graph()['nodes'] == original['nodes']
    assert check.graph()['edges'] == original['edges']

    # Guided connection selection remains available in the stage overview.
    check.tab('Add')
    marker = page.locator('.node-insert').last
    marker.click()
    check.tab('Inspect')
    expect(page.get_by_label('Connection source', exact=True)).to_be_visible()
    expect(page.get_by_label('Connection destination port', exact=True)).to_be_visible()
    page.get_by_role('button', name='Insert a layer', exact=True).click()
    expect(page.get_by_label('Insertion destination')).not_to_have_value('')
    page.get_by_label('Search layer palette').focus()
    page.keyboard.press('Escape')
    expect(page.get_by_label('Insertion destination')).to_have_count(0)
    expect(page.get_by_role('tab', name='Inspect', exact=True)).to_have_attribute('aria-selected', 'true')

    summaries = []
    for preset in ['paper-cnn', 'paper-lstm', 'paper-quaternion', 'residual-tcn', 'tslib-patchtst']:
        check.tab('Architecture')
        page.get_by_label('Load graph preset').select_option(preset)
        graph = check.graph(lambda g: g.get('preset_id') == preset)
        page.wait_for_timeout(800)
        expect(page.locator('.architecture-node-shape').first).to_be_visible()
        assert page.locator('.architecture-diagram .react-flow').count() == 1
        assert not check.errors, check.errors
        if preset == 'paper-cnn':
            assert_fit(page, 6)
            expect(page.locator('.architecture-node-title strong')).not_to_have_count(0)
            assert page.locator('.disconnected-label').count() == 0
        if preset == 'tslib-patchtst':
            assert_fit(page, 3)
            expect(page.locator('.architecture-node-title strong')).not_to_have_count(0)
        payload = {'architecture': graph, 'window': 10, 'horizon': 1}
        before = page.request.post(f'{url}/api/v1/architectures/validate', data=payload).json()
        payload['architecture'] = {**graph, 'view': {'renderer': 'flow', 'version': 1, 'positions': {}, 'collapsed': [g['id'] for g in graph['groups']], 'camera': {'panX': 2, 'panY': 8, 'zoom': .5}}}
        after_validation = page.request.post(f'{url}/api/v1/architectures/validate', data=payload).json()
        assert before['parameters'] == after_validation['parameters']
        assert before['graph_nodes'] == after_validation['graph_nodes']
        summaries.append((preset, len(graph['nodes']), before['parameters']))
        page.screenshot(path=f'/tmp/hypercast-node-{preset}.png')
        if preset == 'paper-quaternion':
            hyper_group = next(g['id'] for g in graph['groups'] if 'hyper' in g['label'])
            check.find(next(n['id'] for n in graph['nodes'] if n.get('group') == hyper_group))
            page.get_by_role('button', name='Inspect weight structure', exact=True).click()
            expect(page.get_by_role('button', name='Close weight inspector')).to_be_visible()
            weight_zoom = diagram.get_attribute('data-zoom')
            page.get_by_role('button', name='Close weight inspector').click()
            expect(diagram).to_have_attribute('data-zoom', weight_zoom)
    # Expand every actual stage in the larger transformer and render its real branches.
    page.get_by_role('tab', name='Architecture', exact=True).click()
    page.get_by_label('Search layers', exact=True).fill('')
    collapsed_stages = page.locator('.architecture-stage-row button[aria-expanded=false]')
    while collapsed_stages.count():
        collapsed_stages.first.click()
    page.get_by_role('button', name='Fit diagram', exact=True).click()
    page.wait_for_timeout(2000)
    page.screenshot(path='/tmp/hypercast-node-expanded-transformer.png')
    assert float(diagram.get_attribute('data-zoom')) < .25, 'Fit should show the full expanded transformer'
    assert not check.errors, check.errors
    assert not check.jobs, check.jobs

    assert page.locator('.architecture-diagram canvas').count() == 0
    print('PASS: node selection, exact connections, sidebar opening, pan/zoom/fit/reset, resize, navigation, weight inspector and expanded transformer.')
    print('Unchanged validation and parameter counts:', summaries)


def drag(page, source, target):
    page.mouse.move(source['x'], source['y'])
    page.mouse.down()
    page.mouse.move(target['x'], target['y'], steps=15)
    page.mouse.up()
    page.wait_for_timeout(450)


def center(locator):
    box = locator.bounding_box()
    return {'x': box['x'] + box['width'] / 2, 'y': box['y'] + box['height'] / 2}


def assert_fit(page, count):
    cards = page.locator('.architecture-node')
    expect(cards).to_have_count(count)
    box = page.locator('.architecture-diagram').bounding_box()
    for card in cards.all():
        b = card.bounding_box()
        assert b['x'] >= box['x'] and b['x'] + b['width'] <= box['x'] + box['width'], (box, b)


def check_fixed_layout(page, url):
    check = EditorCheck(page)
    page.goto(url)
    original = check.graph()
    page.wait_for_timeout(800)
    assert_fit(page, 3)
    cards = page.locator('.architecture-node')
    cards.first.locator('.architecture-node-title strong').click()
    cards.nth(1).locator('.architecture-node-title strong').click(modifiers=['Shift'])
    expect(page.locator('.architecture-node[aria-pressed=true]')).to_have_count(2)
    cards.first.focus()
    cards.first.press('Enter')
    expect(page.locator('.architecture-node[aria-pressed=true]')).to_have_count(1)
    card = cards.first
    node_id = card.get_attribute('data-block-id')
    before = card.bounding_box()
    undo_disabled = page.get_by_role('button', name='Undo', exact=True).is_disabled()
    start = center(card.locator('.architecture-node-title strong'))
    drag(page, start, {'x': start['x'] + 80, 'y': start['y'] + 90})
    after = card.bounding_box()
    assert abs(after['x'] - before['x']) < 2 and abs(after['y'] - before['y']) < 2, (before, after)
    assert page.get_by_role('button', name='Undo', exact=True).is_disabled() == undo_disabled
    expect(page.locator('.react-flow__handle.connectable')).to_have_count(0)
    expect(page.locator('.react-flow__edgeupdater')).to_have_count(0)
    assert check.graph()['nodes'] == original['nodes'] and check.graph()['edges'] == original['edges']
    check.tab('Architecture')
    page.get_by_role('button', name='Save graph', exact=True).click()
    expect(page.get_by_text('Graph and canvas layout saved.', exact=True)).to_be_visible()
    saved = check.saved[-1]
    assert saved['view']['positions'] == {}
    assert saved['nodes'] == original['nodes'] and saved['edges'] == original['edges']

    # Both the former freeform layout and Three.js camera migrate to automatic placement.
    for renderer in ['flow', 'three']:
        legacy = {**saved, 'view': {**saved['view'], 'renderer': renderer, 'positions': {node_id: {'x': 99999, 'y': -99999}}, 'camera': {'panX': 99999, 'panY': 99999, 'zoom': .025}}}
        validation_count = len(check.submitted)
        page.get_by_label('Import architecture').set_input_files({'name': 'legacy.json', 'mimeType': 'application/json', 'buffer': json.dumps(legacy).encode()})
        loaded = check.graph(lambda _: len(check.submitted) > validation_count)
        page.wait_for_timeout(500)
        assert_fit(page, 3)
        assert loaded['nodes'] == original['nodes'] and loaded['edges'] == original['edges']
    assert not check.errors, check.errors
    assert not check.jobs, check.jobs
    print('PASS: fixed nodes resist dragging, keyboard/multiple selection, passive ports, automatic layout saving, and legacy freeform/Three.js migration.')


def check_mobile(browser, url):
    context = browser.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
    page = context.new_page()
    check = EditorCheck(page)
    page.goto(url)
    check.graph()
    expect(page.get_by_role('button', name='Show left panel')).to_be_visible()
    page.get_by_role('button', name='Show left panel').click()
    expect(page.get_by_role('complementary', name='Workspace controls')).to_be_visible()
    page.get_by_role('button', name='Hide left panel').click()
    expect(page.get_by_role('button', name='Fit diagram', exact=True)).to_be_visible()
    diagram = page.locator('.architecture-diagram')
    before = float(diagram.get_attribute('data-zoom'))
    page.get_by_role('button', name='Zoom in canvas', exact=True).click()
    assert float(diagram.get_attribute('data-zoom')) > before
    page.screenshot(path='/tmp/hypercast-node-mobile.png')
    assert not check.errors, check.errors
    context.close()
    print('PASS: narrow-screen overlay, accessible icon controls and zoom.')


if __name__ == '__main__':
    options = args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': options.browser_binary} if options.browser_binary else {}))
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        check_view(page, options.url)
        page.close()
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        check_overview(page, options.url)
        page.close()
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        check_fixed_layout(page, options.url)
        page.close()
        check_mobile(browser, options.url)
        browser.close()
