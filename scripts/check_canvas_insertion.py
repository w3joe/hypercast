"""Node editor guided-editing regression against the local playground.

Requires Playwright + Chromium. Save requests are intercepted; training is blocked.
Usage: python scripts/check_canvas_insertion.py [url] [--browser-binary PATH]
"""
import argparse
import copy
import json
from playwright.sync_api import expect, sync_playwright


def edge_id(edge):
    return json.dumps([edge['source'], edge['target'], edge['port']], separators=(',', ':'))


class EditorCheck:
    def __init__(self, page):
        self.page, self.submitted, self.errors, self.saved, self.jobs = page, [], [], [], []
        page.on('pageerror', lambda error: self.errors.append(str(error)))
        page.on('request', self.observe)
        page.route('**/api/v1/architectures', self.save)
        page.route('**/api/v1/jobs', self.job)

    def observe(self, request):
        if request.url.endswith('/architectures/validate') and request.method == 'POST':
            self.submitted.append(request.post_data_json['architecture'])

    def save(self, route):
        if route.request.method != 'POST':
            route.continue_()
            return
        payload = route.request.post_data_json
        self.saved.append(payload)
        route.fulfill(json={'id': 'browser-test-only', 'spec': payload, 'view': payload['view']})

    def job(self, route):
        if route.request.method == 'POST':
            self.jobs.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    def graph(self, predicate=lambda _: True):
        for _ in range(150):
            self.page.wait_for_timeout(100)
            if self.submitted and predicate(self.submitted[-1]):
                return copy.deepcopy(self.submitted[-1])
        raise AssertionError('Expected executable graph was not validated')

    def tab(self, name):
        self.page.get_by_role('tab', name=name, exact=True).click()

    def find(self, node_id):
        self.tab('Architecture')
        self.page.get_by_label('Search layers', exact=True).fill(node_id)
        # The tree searches IDs as well as human labels.
        self.page.locator('.architecture-tree').get_by_role('button', name=f'Select {node_id}', exact=True).click()
        expect(self.page.get_by_role('tab', name='Inspect', exact=True)).to_have_attribute('aria-selected', 'true')

    def insert(self, edge, label):
        self.tab('Add')
        self.page.get_by_label('Insertion destination', exact=True).select_option(edge_id(edge))
        self.page.get_by_label('Search layer palette').fill(label)
        self.page.get_by_title(f'Add {label}', exact=True).click()


def check_insertion(page, url):
    check = EditorCheck(page)
    page.goto(url)
    expect(page.locator('.architecture-diagram .react-flow')).to_have_count(1)
    original = check.graph()
    output_edge = next(e for e in original['edges'] if e['target'] == original['output'])
    check.insert(output_edge, 'Dropout')
    inserted = check.graph(lambda g: len(g['nodes']) == len(original['nodes']) + 1)
    new_id = next(n['id'] for n in inserted['nodes'] if n['id'] not in {n['id'] for n in original['nodes']})
    assert {'source': output_edge['source'], 'target': new_id, 'port': 'x'} in inserted['edges']
    assert {**output_edge, 'source': new_id} in inserted['edges']
    assert output_edge not in inserted['edges']
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['edges'] == original['edges'])
    page.get_by_role('button', name='Redo', exact=True).click()
    check.graph(lambda g: g['edges'] == inserted['edges'])

    needed, pending = set(), [original['output']]
    while pending:
        node = pending.pop()
        if node in needed:
            continue
        needed.add(node)
        pending.extend(e['source'] for e in original['edges'] if e['target'] == node)
    destination = next(e for e in original['edges'] if e['target'] in needed and e['target'] != original['output'] and e['source'] != output_edge['source'])
    check.find(new_id)
    page.get_by_role('button', name='Advanced tools', exact=True).click()
    page.get_by_role('button', name='Move selected layer', exact=True).click()
    page.get_by_label('Insertion destination').select_option(edge_id(destination))
    moved = check.graph(lambda g: {'source': destination['source'], 'target': new_id, 'port': 'x'} in g['edges'])
    assert len(moved['nodes']) == len(inserted['nodes'])
    assert output_edge in moved['edges']
    assert {**destination, 'source': new_id} in moved['edges']
    # The palette explains multi-input layers before an invalid insertion.
    check.tab('Add')
    page.get_by_label('Insertion destination').select_option(edge_id(output_edge))
    page.get_by_label('Search layer palette').fill('Residual / add')
    expect(page.get_by_title('Add Residual / add', exact=True)).to_be_disabled()
    assert check.graph()['edges'] == moved['edges']

    check.find(new_id)
    page.get_by_role('button', name='Delete selected', exact=True).click()
    deleted = check.graph(lambda g: not any(n['id'] == new_id for n in g['nodes']))
    assert not any(e['source'] == new_id or e['target'] == new_id for e in deleted['edges'])
    assert destination not in deleted['edges']  # Deletion must not heal wiring.
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: g['edges'] == moved['edges'])

    # Independent disconnected draft, explicit sources, replacement, and invalid drafts.
    check.tab('Add')
    page.get_by_label('Insertion destination').select_option('')
    page.get_by_label('Search layer palette').fill('Dense')
    expect(page.get_by_title('Add Dense', exact=True)).to_be_disabled()
    page.locator('.advanced-add summary').click()
    page.get_by_role('button', name='Create unconnected layer', exact=True).click()
    page.get_by_title('Add Dense', exact=True).click()
    draft = check.graph(lambda g: len(g['nodes']) == len(moved['nodes']) + 1)
    dense_id = next(n['id'] for n in draft['nodes'] if n['id'] not in {n['id'] for n in moved['nodes']})
    assert not any(e['target'] == dense_id for e in draft['edges'])
    expect(page.locator('.architecture-node-warning')).not_to_have_count(0)
    if page.locator('.canvas-input-picker').get_attribute('open') is None:
        page.get_by_text('Input connections', exact=True).click()
    page.get_by_label('Source for x', exact=True).select_option(output_edge['source'])
    connected = check.graph(lambda g: any(e['target'] == dense_id for e in g['edges']))
    page.get_by_label('Layer type', exact=True).select_option('dropout')
    replaced = check.graph(lambda g: not any(n['id'] == dense_id for n in g['nodes']))
    assert len(replaced['nodes']) == len(connected['nodes'])
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: any(n['id'] == dense_id for n in g['nodes']))
    check.find(new_id)
    rate = page.get_by_label('p', exact=True)
    rate.fill('2')
    page.wait_for_timeout(450)
    assert next(n for n in check.graph()['nodes'] if n['id'] == new_id)['params']['p'] != 2
    rate.press('Enter')
    check.graph(lambda g: next(n for n in g['nodes'] if n['id'] == new_id)['params']['p'] == 2)
    expect(page.get_by_role('alert').first).to_be_visible()
    check.tab('Run')
    expect(page.get_by_role('button', name='Run validation', exact=True)).to_be_disabled()
    page.get_by_role('button', name='Undo', exact=True).click()
    check.graph(lambda g: next(n for n in g['nodes'] if n['id'] == new_id)['params']['p'] != 2)
    expect(page.get_by_role('button', name='Run validation', exact=True)).to_be_enabled(timeout=15000)

    check.tab('Architecture')
    page.get_by_role('button', name='Save graph', exact=True).click()
    expect(page.get_by_text('Graph and canvas layout saved.', exact=True)).to_be_visible()
    assert check.saved[-1]['view']['renderer'] == 'flow'
    assert check.saved[-1]['view']['version'] == 1
    assert check.saved[-1]['view']['positions'] == {}
    assert check.saved[-1]['view']['camera']['zoom'] > 0
    assert not check.jobs, check.jobs
    assert not check.errors, check.errors
    print('PASS: insert, move, reject multi-input insertion, undo/redo, delete without inferred wiring, disconnected drafts, reconnect, replace, blur/Enter commits, invalid-draft training guard, versioned save (intercepted).')
    return check


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', nargs='?', default='http://127.0.0.1:8765')
    parser.add_argument('--browser-binary')
    return parser.parse_args()


if __name__ == '__main__':
    options = args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': options.browser_binary} if options.browser_binary else {}))
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        check_insertion(page, options.url)
        page.screenshot(path='/tmp/hypercast-node-editing.png')
        browser.close()
