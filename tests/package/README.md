# Verify package build and install

When developing a package, it's important that this package can be build,
installed, and that the expected metadata is set. This allows changes to the
build system to be made with confidence. This test module verifies that the
package can be installed, and that the version attribute matches the value of
`__version__`.
