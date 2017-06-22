soap3_results = {
    'seidel': {
        'analysis_duration': 2.2875101566314697,
        'original': {
            'area': 411,
            'error': 2.175569875362271e-07,
            'expression': '((((a + b) + c) + d) + e) * 0.2'
        },
        'analysis': [
            {
                'area': 1058,
                'error': 1.7136335372924805e-07,
                'expression': '(((d * 0.2) + (a * 0.2)) + ((c * 0.2) + (e * 0.2))) + (b * 0.2)'
            },
            {
                'area': 544,
                'error': 2.0712616333184997e-07,
                'expression': '((((a + c) + d) + b) * 0.2) + (e * 0.2)'
            },
            {
                'area': 411,
                'error': 2.175569875362271e-07,
                'expression': '((((a + c) + b) + d) + e) * 0.2'
            },
            {
                'area': 649,
                'error': 1.9371512394172896e-07,
                'expression': '((c + e) + ((b + d) + a)) * 0.2'
            },
            {
                'area': 554,
                'error': 1.8179417793362518e-07,
                'expression': '((a + (c + e)) * 0.2) + ((d * 0.2) + (b * 0.2))'
            },
            {
                'area': 564,
                'error': 1.8030405612989853e-07,
                'expression': '((c + e) * 0.2) + (((d * 0.2) + (a * 0.2)) + (b * 0.2))'
            },
            {
                'area': 935,
                'error': 1.7136336794010276e-07,
                'expression': '((c * 0.2) + (e * 0.2)) + (((d * 0.2) + (b * 0.2)) + (a * 0.2))'
            }
        ],
        'vary_precision': [
            {
                'area': 411,
                'error': 2.175569875362271e-07,
                'expression': '((((a + b) + c) + d) + e) * 0.2'
            },
        ],
        'loop': {
            'original': {
                'area': 603,
                'error': 1.0681659659894649e-05,
            },
            'analysis': {
            },
            'vary_precision': [
                # all areas come out the same unless Virtex6 luts are used
                { # 21
                    'area': 603, # 3303
                    'error': 4.272656224202365e-05,
                },
                { # 22
                    'area': 603, # 3417
                    'error': 2.282651257701218e-05,
                },
                { # 23
                    'area': 603, # 3518
                    'error': 1.0681659659894649e-05,
                },
                { # 24
                    'area': 603, # 3707
                    'error': 5.706639285563142e-06,
                },
                { # 25
                    'area': 603, # 3785
                    'error': 2.6704132096710964e-06,
                },
            ],
        }
    }
}
