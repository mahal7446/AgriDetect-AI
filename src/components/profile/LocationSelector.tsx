import { useState, useEffect, useRef } from "react";
import { MapPin, Search, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { searchDistricts, formatLocation, District } from "@/data/indianCities";

interface LocationSelectorProps {
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
}

export const LocationSelector = ({ value, onChange, disabled = false }: LocationSelectorProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [filteredDistricts, setFilteredDistricts] = useState<District[]>([]);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Handle search
    useEffect(() => {
        if (searchQuery.trim()) {
            const results = searchDistricts(searchQuery);
            setFilteredDistricts(results);
        } else {
            setFilteredDistricts([]);
        }
    }, [searchQuery]);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener("mousedown", handleClickOutside);
        }

        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [isOpen]);

    const handleSelect = (district: District) => {
        const formattedLocation = formatLocation(district);
        onChange(formattedLocation);
        setSearchQuery("");
        setIsOpen(false);
    };

    const handleClear = () => {
        onChange("");
        setSearchQuery("");
        setIsOpen(false);
    };

    const handleFocus = () => {
        if (!disabled) {
            setIsOpen(true);
        }
    };

    return (
        <div className="space-y-2" ref={dropdownRef}>
            <Label htmlFor="location">Address (District, State)</Label>
            <div className="relative">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground z-10" />

                {disabled ? (
                    // Read-only mode - just show the value
                    <Input
                        id="location"
                        value={value || "No address set"}
                        disabled={true}
                        className="pl-9 bg-muted"
                    />
                ) : (
                    // Edit mode - searchable input
                    <>
                        <Input
                            ref={inputRef}
                            id="location"
                            value={isOpen ? searchQuery : value}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onFocus={handleFocus}
                            placeholder="Search for your district..."
                            className="pl-9 pr-20"
                        />
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                            {value && (
                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    onClick={handleClear}
                                    className="h-7 px-2"
                                >
                                    <X className="w-4 h-4" />
                                </Button>
                            )}
                            <Search className="w-4 h-4 text-muted-foreground" />
                        </div>
                    </>
                )}

                {/* Dropdown */}
                {isOpen && !disabled && (
                    <div className="absolute z-50 w-full mt-1 bg-popover border rounded-md shadow-lg max-h-[300px] overflow-y-auto">
                        {searchQuery.trim() === "" ? (
                            <div className="p-4 text-sm text-muted-foreground text-center">
                                Start typing to search for your district...
                            </div>
                        ) : filteredDistricts.length > 0 ? (
                            <div className="py-1">
                                {filteredDistricts.map((district, index) => (
                                    <button
                                        key={`${district.name}-${district.state}-${index}`}
                                        onClick={() => handleSelect(district)}
                                        className="w-full px-4 py-2 text-left hover:bg-accent transition-colors flex items-start gap-3 group"
                                    >
                                        <MapPin className="w-4 h-4 mt-0.5 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0" />
                                        <div className="flex-1">
                                            <div className="font-medium text-sm">{district.name}</div>
                                            <div className="text-xs text-muted-foreground">{district.state}</div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <div className="p-4 text-sm text-muted-foreground text-center">
                                No districts found for "{searchQuery}"
                            </div>
                        )}
                    </div>
                )}
            </div>

            {!disabled && (
                <p className="text-xs text-muted-foreground">
                    Search and select your district from the dropdown
                </p>
            )}
        </div>
    );
};
