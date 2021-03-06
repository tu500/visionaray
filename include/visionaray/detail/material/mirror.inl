// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T>
VSNRAY_FUNC
inline spectrum<T> mirror<T>::ambient() const
{
    return spectrum<T>(0.0); // TODO: no support for  ambient
}

template <typename T>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> mirror<T>::shade(SR const& sr) const
{
    return specular_brdf_.f(sr.normal, sr.view_dir, sr.light_dir);
}

template <typename T>
template <typename SR, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> mirror<T>::sample(
        SR const&       sr,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Sampler&        sampler
        ) const
{
    return specular_brdf_.sample_f(sr.normal, sr.view_dir, refl_dir, pdf, sampler);
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_cr(spectrum<T> const& cr)
{
    specular_brdf_.cr = cr;
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> mirror<T>::get_cr() const
{
    return specular_brdf_.cr;
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_kr(T const& kr)
{
    specular_brdf_.kr = kr;
}

template <typename T>
VSNRAY_FUNC
inline T mirror<T>::get_kr() const
{
    return specular_brdf_.kr;
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_ior(spectrum<T> const& ior)
{
    specular_brdf_.ior = ior;
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_ior(T ior)
{
    specular_brdf_.ior = spectrum<T>(ior);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> mirror<T>::get_ior() const
{
    return specular_brdf_.ior;
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_absorption(spectrum<T> const& absorption)
{
    specular_brdf_.absorption = absorption;
}

template <typename T>
VSNRAY_FUNC
inline void mirror<T>::set_absorption(T absorption)
{
    specular_brdf_.absorption = spectrum<T>(absorption);
}

template <typename T>
VSNRAY_FUNC
inline spectrum<T> mirror<T>::get_absorption() const
{
    return specular_brdf_.absorption;
}

} // visionaray
