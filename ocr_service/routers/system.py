"""
This module contains the routers for the system endpoints.

    
    return ReconStatusResponse(
        reconstruction_enabled=curr_settings.enable_reconstruction,
        package_installed=CapabilityProvider.is_reconstruction_available(),
        package_version=CapabilityProvider.get_reconstruction_version(),
    )