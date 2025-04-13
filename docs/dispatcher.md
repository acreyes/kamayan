# Dispatcher

Kernels in kamayan will dispatch into functions, often based on
compile time template parameters. This allows for modularity
in the code/solver capabilities without code duplication, and
avoids additional register usage, particularly in GPU kernels,
if a runtime check needs to be performed at the call site.

```cpp title="physics/hydro/hydro_add_flux_tasks.cpp:rea"
--8<-- "physics/hydro/hydro_add_flux_tasks.cpp:rea"
```

In the above snippet the `Reconstruct` and `RiemannFlux` functions
will dispatch to the correct methods depending on the template
parameters `recon` & `riemann`.

Having these template parameters makes for concise code in the 
kernels, but as the number of parameters used in a function increases
the combinatorial combinations that need to be explicitly instantiated,
if we want the choices to be runtime decisions. Kamayan provides a dispatcher
that simplifies the dispatching of templated functions from an arbitrary
number of runtime parameters.

```cpp title="dispatcher/tests/test_dispatcher.cpp:functor"
--8<-- "dispatcher/tests/test_dispatcher.cpp:functor"
```

Such functions can be provided as a functor that has two `using decl`s:

* `options` -- defines a `OptTypeList` that describes how the functor
can be built from non-type template parameter `enum class`s, and their possible
runtime values.

* `value` -- the return type of the function to be dispatched

The dispatcher will template and call the contained `dispatch` function,
forwarding any arguments that it is passed. 

```cpp title="dispatcher/tests/test_dispatcher.cpp:dispatch"
--8<-- "dispatcher/tests/test_dispatcher.cpp:dispatch"
```

## Options

`enum class`s that can be used as template parameters are declared with the 
`POLYMORPHIC_PARM` macro defined in `dispatcher/options.hpp`.

```cpp title="dispatcher/tests/test_dispatcher.cpp:poly"
--8<-- "dispatcher/tests/test_dispatcher.cpp:poly"
```

Only `enum`s that have been declared like this can be used in the `options` list
used by the dispatcher. The `options` `OptTypeList` is a list of types, one for 
each template parameter in the `dispatch` function, that are used to
instantiate the template parameters. It can be helpful to mentally map the 
`OptList` types onto a switch statement for the template parameter.

```cpp
// OptList<Foo, Foo::a, Foo::b> roughly corresponds to
template<typename... Args>
void Dispatcher<MyFunctor>(Foo foo_v, Args ...&&args) {
   switch (foo_v) {
      case Foo::a :
         MyFunctor.template <Foo::a>(std::forward<Args>(args)...);
         break;
      case Foo::b :
         MyFunctor.template <Foo::b>(std::forward<Args>(args)...);
         break;
   }
}
```

## Composite Types

Another common pattern with the non-type template parameters we have discussed
so far is to use them to build more complex type traits as composite options
built from our `POLYMORPHIC_PARM`s.

```cpp title="dispatcher/tests/test_dispatcher.cpp"
--8<-- "dispatcher/tests/test_dispatcher.cpp:comp-op"

--8<-- "dispatcher/tests/test_dispatcher.cpp:composite"
```

The functors now look very similar to the ones we looked at before. The main
difference is that we need to provide a factory type, `CompositeFactory`, 
that informs the dispatcher how to build our composite type from
the provided runtime options.

```cpp title="dispatcher/tests/test_dispatcher.cpp:factory"
--8<-- "dispatcher/tests/test_dispatcher.cpp:factory"
```

The factory interface is very similar to the previous functor definition, with
three `using decl` definitions

* `options` -- and `OptTypeList` like in the functors to enumerate the runtime options
and their allowed values.
* `composite` -- a templated `using decl` that aliases the type the factory produces
* `type` -- always points to itself

Additionally the factory needs to inherit from the `OptionFactory` type to inform
the dispatcher that this is a composite type.

```cpp title="dispatcher/tests/test_dispatcher.cpp:comp-disp"
--8<-- "dispatcher/tests/test_dispatcher.cpp:comp-disp"
```

## Using `Config` for dispatch

Instead of passing each individual runtime template parameter value to the dispatcher
the global kamayan `Config` may also be used. The dispatcher will instead pull the
runtime values directly from `POLYMORPHIC_PARMS` that have been registered to the `Config`

```cpp title="physics/hydro/primconsflux.cpp"
--8<-- "physics/hydro/primconsflux.cpp:impl"

--8<-- "physics/hydro/primconsflux.cpp:prepare-cons"
```

