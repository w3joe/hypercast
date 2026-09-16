"""Compare-page acceptance checks against local saved results; all job writes blocked."""
import csv
from pathlib import Path
from statistics import mean
from playwright.sync_api import sync_playwright, expect
from check_canvas_insertion import args


def check(page, url):
    errors, writes, forecast_requests = [], [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: forecast_requests.append(request.url) if '/forecasts?' in request.url else None)
    def block_jobs(route):
        if route.request.method != 'GET':
            writes.append(route.request.url); route.abort()
        else:
            route.continue_()
    page.route('**/api/v1/jobs**', block_jobs)
    archived = page.request.get(url + '/api/v1/comparison-archives').json()
    assert len(archived) == 120 and len({r['archive']['backbone'] for r in archived}) == 15
    assert sum(len(r['runs']) for r in archived) == 360
    page.goto(url)
    page.get_by_role('button', name='Compare', exact=True).click()
    expect(page.get_by_label('Comparison stage')).to_have_value('final_test', timeout=15000)
    expect(page.get_by_text('15 backbones · 8 variants · 3 seeds', exact=True)).to_be_visible()
    expect(page.get_by_role('heading', name='How error changes further ahead')).to_be_visible()
    expect(page.locator('.research-comparison .chart-panel')).to_have_count(3)
    tradeoff = page.get_by_role('article', name='Does a larger model earn its size?')
    expect(tradeoff).to_be_visible()
    expect(tradeoff.locator('.parameter-point')).to_have_count(3)
    first_point = tradeoff.locator('.parameter-point').first
    first_point.hover()
    expect(tradeoff.get_by_role('status')).to_contain_text(f"{archived[0]['runs'][0]['parameters']:,}")
    first_point.focus()
    first_point.press('Escape')
    expect(tradeoff.get_by_role('status')).to_contain_text('Hover, tap')
    tradeoff.screenshot(path='/tmp/hypercast-parameter-tradeoff.png')
    assert not forecast_requests, 'Optional forecasts must not fetch before expansion'
    score_table = page.locator('.comparison-numbers table')
    for i, run in enumerate(archived[:3]):
        value = mean(r['mae'] / r['persistence_mae'] for r in run['runs'])
        expect(score_table.locator('tbody tr').nth(i).locator('td').first).to_have_text(f'{value:.4f}')
    page.wait_for_timeout(500)
    page.screenshot(path='/tmp/hypercast-compare-final.png')

    picker = page.locator('.topbar-run-picker > summary')
    picker.click()
    variant = page.get_by_label('Comparison variant filter')
    variant.select_option('native')
    expect(page.locator('.topbar-run-menu input[type=checkbox]')).to_have_count(15)
    page.get_by_label('Find comparison runs').fill('FiLM')
    expect(page.locator('.topbar-run-menu input[type=checkbox]')).to_have_count(1)
    page.locator('.topbar-run-menu input[type=checkbox]').check()
    variant.select_option('quaternion')
    page.locator('.topbar-run-menu input[type=checkbox]').check()
    picker.click()
    expect(score_table.locator('tbody tr')).to_have_count(5)
    expect(score_table.get_by_text('FiLM · native', exact=True)).to_be_visible()
    expect(score_table.get_by_text('FiLM · quaternion', exact=True)).to_be_visible()
    expect(tradeoff.locator('.parameter-point')).to_have_count(5)
    real_shapes = tradeoff.locator('.parameter-point').evaluate_all("els => els.map(el => ({ name: el.getAttribute('aria-label'), shape: el.lastElementChild.tagName, color: el.lastElementChild.getAttribute('fill') }))")
    film = [r for r in real_shapes if r['name'].startswith('FiLM')]
    assert len(film) == 2 and film[0]['color'] == film[1]['color']
    assert {r['shape'] for r in film} == {'circle', 'path'}

    page.get_by_label('Comparison metric').select_option('mse_ratio')
    first = archived[0]
    expected = mean(r['mse'] / r['persistence_mse'] for r in first['runs'])
    expect(score_table.locator('tbody tr').first.locator('td').first).to_have_text(f'{expected:.4f}')
    with page.expect_download() as download:
        page.get_by_role('button', name='Export scores & diagnostics').click()
    target = Path('/tmp/hypercast-compare-check.csv'); download.value.save_as(target)
    exported = list(csv.DictReader(target.open()))
    assert len(exported) == 5 and {r['stage'] for r in exported} == {'final_test'}
    assert {r['included'] for r in exported} == {'3'}
    assert abs(float(exported[0]['mean']) - expected) <= 1e-12 * max(1, abs(expected))
    page.get_by_role('tab', name='Diagnostics', exact=True).click()
    expect(score_table.get_by_role('columnheader', name='Mean replicate P95 error', exact=True)).to_be_visible()
    assert score_table.locator('tbody').inner_text().count('—') >= 5  # No invented direction or large-move scores.
    page.get_by_role('tab', name='Settings', exact=True).click()
    expect(score_table.get_by_text('test: rows 11520–14399', exact=False).first).to_be_visible()

    page.locator('.comparison-forecast-disclosure > summary').click()
    expect(page.locator('.forecast-replay .replay-chart')).to_have_count(1, timeout=15000)
    expect(page.locator('.research-comparison .chart-panel')).to_have_count(4)
    expect(page.get_by_role('heading', name='Signed error', exact=True)).to_have_count(0)
    page.get_by_label('Replay lead').select_option('5')
    expect(page.get_by_text('Forecast replay / lead 5', exact=True)).to_be_visible()
    page.locator('.comparison-forecast-disclosure > summary').click()
    expect(page.locator('.forecast-replay')).to_have_count(0)

    page.get_by_label('Comparison stage').select_option('validation')
    expect(page.get_by_label('Comparison evaluation group').locator('option:checked')).to_contain_text('18 runs')
    page.get_by_role('tab', name='Scores', exact=True).click()
    before = score_table.locator('tbody th').all_text_contents()
    page.get_by_label('Comparison window').select_option('20')
    expect(page.get_by_label('Comparison horizon')).to_have_value('5')
    assert score_table.locator('tbody th').all_text_contents() == before
    expect(page.get_by_role('heading', name='How scores change with the task')).to_be_visible()
    page.get_by_label('Comparison evaluation group').select_option(label=next(t for t in page.get_by_label('Comparison evaluation group').locator('option').all_text_contents() if 'robust' in t))
    expect(page.get_by_text('Only one run uses this split plan.', exact=False)).to_be_visible()
    expect(score_table.locator('tbody tr')).to_have_count(1)
    page.get_by_label('Comparison horizon').select_option('10')
    page.get_by_label('Comparison stage').select_option('quick')
    expect(page.get_by_text('Quick checks verify execution; their scores are preliminary.', exact=True)).to_be_visible()
    page.get_by_label('Comparison stage').select_option('final_test')
    expect(score_table.locator('tbody tr')).to_have_count(5)  # Selection survived evaluation changes.

    page.set_viewport_size({'width': 390, 'height': 844})
    if page.get_by_role('button', name='Hide left panel', exact=True).is_visible():
        page.get_by_role('button', name='Hide left panel', exact=True).click()
    page.locator('.app-content').evaluate('el => { el.scrollTop = 0 }')
    expect(page.get_by_role('heading', name='Compare evaluations')).to_be_visible()
    assert page.locator('.research-comparison').evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    page.screenshot(path='/tmp/hypercast-compare-mobile.png')
    tradeoff.scroll_into_view_if_needed()
    assert tradeoff.evaluate('el => el.scrollWidth <= el.clientWidth + 1')
    tradeoff.locator('.parameter-point').first.focus()
    expect(tradeoff.get_by_role('status')).to_contain_text('parameters')
    tradeoff.screenshot(path='/tmp/hypercast-parameter-tradeoff-mobile.png')
    assert not writes, writes
    assert not errors, errors
    print('PASS: 15 backbones / 120 archived variants / 360 fits, exact scores and export, variant search, read-only final results, 3+1 charts, parameter scatter hover/keyboard/colours/shapes, aligned forecast overlay, numeric diagnostics/settings, evaluation navigation and mobile layout.')


if __name__ == '__main__':
    options = args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': options.browser_binary} if options.browser_binary else {}))
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        check(page, options.url)
        browser.close()
