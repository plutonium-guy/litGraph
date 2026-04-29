"""SitemapLoader — fetch sitemap.xml + crawl all URLs through WebLoader.
Standard pattern for docs-site ingestion (Read the Docs, Hugo, Jekyll)."""
import http.server
import threading

from litgraph.loaders import SitemapLoader


class _RoutedFake(http.server.BaseHTTPRequestHandler):
    """Routes requests to canned responses keyed by path."""
    ROUTES: dict = {}

    def do_GET(self):
        body, status, ctype = self.ROUTES.get(
            self.path, ("not found", 404, "text/plain"),
        )
        out = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(routes):
    _RoutedFake.ROUTES = routes
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RoutedFake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _sitemap_xml(urls):
    entries = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + entries + "</urlset>"
    )


def _index_xml(child_urls):
    entries = "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in child_urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + entries + "</sitemapindex>"
    )


def test_loads_pages_from_urlset_sitemap():
    # Bind first to discover the port, build sitemap with that port,
    # then start the server. (Race-y but stable enough for unit tests.)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    sitemap = _sitemap_xml([
        f"http://127.0.0.1:{port}/page1",
        f"http://127.0.0.1:{port}/page2",
    ])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = {
        "/sitemap.xml": (sitemap, 200, "application/xml"),
        "/page1": ("<p>page one body</p>", 200, "text/html"),
        "/page2": ("<p>page two body</p>", 200, "text/html"),
    }
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(f"http://127.0.0.1:{port}/sitemap.xml")
        docs = loader.load()
        assert len(docs) == 2
        contents = sorted(d["content"] for d in docs)
        assert "<p>page one body</p>" in contents
        assert "<p>page two body</p>" in contents
        # Provenance metadata.
        for d in docs:
            assert d["metadata"]["sitemap_source"] == f"http://127.0.0.1:{port}/sitemap.xml"
    finally:
        srv.shutdown()


def test_include_pattern_filters_urls():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    sitemap = _sitemap_xml([
        f"http://127.0.0.1:{port}/docs/intro",
        f"http://127.0.0.1:{port}/blog/post",
        f"http://127.0.0.1:{port}/docs/api",
    ])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = {
        "/sitemap.xml": (sitemap, 200, "application/xml"),
        "/docs/intro": ("intro", 200, "text/html"),
        "/docs/api": ("api", 200, "text/html"),
        "/blog/post": ("blog", 200, "text/html"),
    }
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(
            f"http://127.0.0.1:{port}/sitemap.xml",
            include_pattern=r"/docs/",
        )
        docs = loader.load()
        assert len(docs) == 2
        for d in docs:
            assert "/docs/" in d["metadata"]["source"]
    finally:
        srv.shutdown()


def test_exclude_pattern_drops_urls():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    sitemap = _sitemap_xml([
        f"http://127.0.0.1:{port}/published/x",
        f"http://127.0.0.1:{port}/draft/y",
    ])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = {
        "/sitemap.xml": (sitemap, 200, "application/xml"),
        "/published/x": ("pub", 200, "text/html"),
        "/draft/y": ("draft", 200, "text/html"),
    }
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(
            f"http://127.0.0.1:{port}/sitemap.xml",
            exclude_pattern=r"/draft/",
        )
        docs = loader.load()
        assert len(docs) == 1
        assert "published" in docs[0]["metadata"]["source"]
    finally:
        srv.shutdown()


def test_max_urls_caps_total():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    sitemap = _sitemap_xml([
        f"http://127.0.0.1:{port}/p{i}" for i in range(5)
    ])
    routes = {"/sitemap.xml": (sitemap, 200, "application/xml")}
    for i in range(5):
        routes[f"/p{i}"] = (f"page {i}", 200, "text/html")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = routes
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(
            f"http://127.0.0.1:{port}/sitemap.xml",
            max_urls=2,
        )
        docs = loader.load()
        assert len(docs) == 2
    finally:
        srv.shutdown()


def test_sitemapindex_follows_child_sitemaps():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    child1 = _sitemap_xml([f"http://127.0.0.1:{port}/p1"])
    child2 = _sitemap_xml([f"http://127.0.0.1:{port}/p2"])
    index = _index_xml([
        f"http://127.0.0.1:{port}/sitemap-1.xml",
        f"http://127.0.0.1:{port}/sitemap-2.xml",
    ])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = {
        "/sitemap.xml": (index, 200, "application/xml"),
        "/sitemap-1.xml": (child1, 200, "application/xml"),
        "/sitemap-2.xml": (child2, 200, "application/xml"),
        "/p1": ("page 1", 200, "text/html"),
        "/p2": ("page 2", 200, "text/html"),
    }
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(f"http://127.0.0.1:{port}/sitemap.xml")
        docs = loader.load()
        assert len(docs) == 2
    finally:
        srv.shutdown()


def test_invalid_include_regex_raises_value_error():
    try:
        SitemapLoader("http://example/sitemap.xml", include_pattern=r"[unclosed")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "regex" in str(e).lower()


def test_http_error_on_sitemap_surfaces():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _RoutedFake)
    _RoutedFake.ROUTES = {"/sitemap.xml": ("not found", 404, "text/plain")}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SitemapLoader(f"http://127.0.0.1:{port}/sitemap.xml")
        try:
            loader.load()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "404" in str(e)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_loads_pages_from_urlset_sitemap,
        test_include_pattern_filters_urls,
        test_exclude_pattern_drops_urls,
        test_max_urls_caps_total,
        test_sitemapindex_follows_child_sitemaps,
        test_invalid_include_regex_raises_value_error,
        test_http_error_on_sitemap_surfaces,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
